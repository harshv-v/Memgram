"""Unified memory read/write API. Every write that creates a semantic memory
goes through dedup: if an embedding-similar memory exists (distance < 0.15),
we REINFORCE it instead of duplicating — this is what makes habit formation
possible (reinforcement_count is the habit signal)."""
import asyncpg

from memgram.memory.embedder import to_pgvector
from memgram.obs import MEMORIES_TOTAL
from memgram.safety import redact_pii
from memgram import settings_store

DEDUP_DISTANCE = 0.15


class MemoryStore:
    def __init__(self, pool: asyncpg.Pool, embedder):
        self.pool = pool
        self.embedder = embedder

    # -- semantic -----------------------------------------------------------
    async def upsert_semantic(
        self, project_id: str, agent_id: str, user_id: str, content: str,
        memory_type: str = "fact", source: str = "extractor",
        emotional_weight: float = 1.0, stability: float = 2.0,
        scope: str = "private", provenance: str | None = None,
    ) -> dict:
        if await settings_store.get_bool(self.pool, project_id, "pii_redact"):
            content = redact_pii(content)
        emb = to_pgvector(await self.embedder.embed(content))  # network call: keep OUT of the tx
        lock_key = f"{project_id}:{user_id}:{agent_id}"
        async with self.pool.acquire() as conn:
            # Serialize check-then-act per (project,user,agent): without this, two
            # concurrent extractions of the same fact both find "nothing similar"
            # and both INSERT -> duplicates. A transaction-scoped advisory lock
            # makes dedup correct now that the worker runs jobs concurrently.
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1)::bigint)", lock_key)
                similar = await conn.fetchrow(
                    """
                    SELECT id, embedding <=> $4::vector AS dist
                    FROM semantic_memories
                    WHERE project_id = $1 AND user_id = $2 AND agent_id = $3
                      AND memory_tier != 'archived'
                    ORDER BY embedding <=> $4::vector
                    LIMIT 1
                    """,
                    project_id, user_id, agent_id, emb,
                )
                if similar and similar["dist"] is not None and similar["dist"] < DEDUP_DISTANCE:
                    row = await conn.fetchrow(
                        """
                        UPDATE semantic_memories SET
                          reinforcement_count = reinforcement_count + 1,
                          stability           = stability * (1 + 0.2 * retention_score),
                          retention_score     = 1.0,
                          emotional_weight    = GREATEST(emotional_weight, $2),
                          last_accessed_at    = NOW(),
                          memory_tier         = CASE WHEN memory_tier = 'fading'
                                                     THEN 'active' ELSE memory_tier END
                        WHERE id = $1
                        RETURNING id, reinforcement_count
                        """,
                        similar["id"], emotional_weight,
                    )
                    MEMORIES_TOTAL.labels(action="reinforced").inc()
                    return {"id": str(row["id"]), "action": "reinforced",
                            "reinforcement_count": row["reinforcement_count"]}
                row = await conn.fetchrow(
                    """
                    INSERT INTO semantic_memories
                      (project_id, agent_id, user_id, content, memory_type, source,
                       embedding, emotional_weight, stability, scope, provenance)
                    VALUES ($1,$2,$3,$4,$5,$6,$7::vector,$8,$9,$10,$11)
                    RETURNING id
                    """,
                    project_id, agent_id, user_id, content, memory_type, source,
                    emb, emotional_weight, stability, scope, provenance,
                )
                MEMORIES_TOTAL.labels(action="created").inc()
                return {"id": str(row["id"]), "action": "created", "reinforcement_count": 1}

    # -- contradiction / supersession --------------------------------------
    async def find_similar_active(
        self, project_id: str, agent_id: str, user_id: str, content: str,
        limit: int = 8, max_distance: float = 0.8,
    ) -> list[dict]:
        """Top-s active, non-superseded memories nearest to `content`, excluding
        exact duplicates (those go through dedup→reinforce) — i.e. integration
        *candidates* for the operation-selection pass. Mem0-style: a bounded
        top-s set, not "everything within a window"; the loose max_distance only
        cuts near-orthogonal noise. Distance nominates; the LLM decides the op."""
        emb = to_pgvector(await self.embedder.embed(content))
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, embedding <=> $4::vector AS dist
                FROM semantic_memories
                WHERE project_id = $1 AND user_id = $2 AND agent_id = $3
                  AND memory_tier != 'archived' AND superseded_by IS NULL
                ORDER BY embedding <=> $4::vector
                LIMIT $5
                """,
                project_id, user_id, agent_id, emb, limit,
            )
        return [{"id": str(r["id"]), "content": r["content"], "dist": float(r["dist"])}
                for r in rows if r["dist"] is not None
                and DEDUP_DISTANCE <= float(r["dist"]) <= max_distance]

    async def update_semantic(self, memory_id: str, content: str) -> dict | None:
        """UPDATE operation: rewrite an existing memory's content in place
        (re-embedding it), preserving its id and reinforcement history. Used when
        a new fact refines/augments the same attribute — the memory gets the
        merged text plus a reinforcement bump, so habit strength is never lost."""
        async with self.pool.acquire() as conn:
            proj = await conn.fetchval(
                "SELECT project_id FROM semantic_memories WHERE id = $1", memory_id)
        if proj and await settings_store.get_bool(self.pool, proj, "pii_redact"):
            content = redact_pii(content)
        emb = to_pgvector(await self.embedder.embed(content))
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE semantic_memories SET
                  content             = $2,
                  embedding           = $3::vector,
                  reinforcement_count = reinforcement_count + 1,
                  stability           = stability * (1 + 0.2 * retention_score),
                  retention_score     = 1.0,
                  last_accessed_at    = NOW(),
                  memory_tier         = CASE WHEN memory_tier = 'fading'
                                             THEN 'active' ELSE memory_tier END
                WHERE id = $1 AND superseded_by IS NULL
                RETURNING id, reinforcement_count
                """,
                memory_id, content, emb,
            )
        if row is None:
            return None
        MEMORIES_TOTAL.labels(action="updated").inc()
        return {"id": str(row["id"]), "action": "updated",
                "reinforcement_count": row["reinforcement_count"]}

    async def supersede(self, old_id: str, new_id: str) -> None:
        """Deprecate a stale memory: link it to its replacement and archive it.
        Archived + superseded means retrieval, dedup, and decay all skip it."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE semantic_memories
                SET superseded_by = $2, memory_tier = 'archived'
                WHERE id = $1
                """,
                old_id, new_id,
            )
        MEMORIES_TOTAL.labels(action="superseded").inc()

    # -- episodic -------------------------------------------------------------
    async def log_episodic(
        self, project_id: str, agent_id: str, user_id: str,
        role: str, content: str, emotional_weight: float = 1.0,
    ) -> str:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO episodic_logs
                  (project_id, agent_id, user_id, role, content, emotional_weight)
                VALUES ($1,$2,$3,$4,$5,$6) RETURNING id
                """,
                project_id, agent_id, user_id, role, content, emotional_weight,
            )
            return str(row["id"])

    # -- transactional ingest (episodic + outbox in ONE commit) ---------------
    async def ingest_turn(
        self, project_id: str, agent_id: str, user_id: str,
        entries: list[tuple[str, str]], jobs: list[tuple[str, dict]],
    ) -> list[str]:
        """Write this turn's episodic entries AND its outbox job intents in one
        transaction — either the turn fully exists (rows + pending jobs) or it
        doesn't. Closes the dual-write gap. Returns the outbox row ids."""
        import json as _json
        outbox_ids: list[str] = []
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for role, content in entries:
                    await conn.execute(
                        """
                        INSERT INTO episodic_logs
                          (project_id, agent_id, user_id, role, content)
                        VALUES ($1,$2,$3,$4,$5)
                        """,
                        project_id, agent_id, user_id, role, content,
                    )
                for job_type, payload in jobs:
                    row = await conn.fetchrow(
                        "INSERT INTO outbox (job_type, payload) VALUES ($1, $2::jsonb) RETURNING id",
                        job_type, _json.dumps(payload),
                    )
                    outbox_ids.append(str(row["id"]))
        return outbox_ids
