"""Unified memory read/write API. Every write that creates a semantic memory
goes through dedup: if an embedding-similar memory exists (distance < 0.15),
we REINFORCE it instead of duplicating — this is what makes habit formation
possible (reinforcement_count is the habit signal)."""
import asyncpg

from memgram.memory.embedder import to_pgvector

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
        scope: str = "project", provenance: str | None = None,
    ) -> dict:
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
                if similar and similar["dist"] < DEDUP_DISTANCE:
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
                return {"id": str(row["id"]), "action": "created", "reinforcement_count": 1}

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
