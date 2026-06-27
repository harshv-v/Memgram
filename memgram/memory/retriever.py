"""Ranked retrieval — the hot-path query from section 7 of the design doc.

ranking = cosine_similarity × retention_score × emotional_weight
Retrieved memories are reinforced (access = rehearsal, like human memory).
Scope filter is applied on every SELECT — this is the Router agent's rule,
enforced as SQL, ready for multi-agent scopes later.
"""
import asyncpg

from memgram.memory.embedder import to_pgvector


class Retriever:
    def __init__(self, pool: asyncpg.Pool, embedder):
        self.pool = pool
        self.embedder = embedder

    async def search(
        self, project_id: str, agent_id: str, user_id: str,
        query: str, limit: int = 5, allowed_scope: str = "project",
        enforce_rls: bool = True,
    ) -> list[dict]:
        emb = to_pgvector(await self.embedder.embed(query))
        async with self.pool.acquire() as conn:
            # Per-request tenant isolation at the engine level (migration 004).
            # Transaction-scoped so it can't leak to the next pooled borrower.
            if enforce_rls:
                tx = conn.transaction()
                await tx.start()
                await conn.execute(
                    "SELECT set_config('app.current_user_id', $1, true)", user_id)
            rows = await conn.fetch(
                """
                SELECT id, content, memory_type, retention_score, memory_tier,
                       reinforcement_count,
                       (1 - (embedding <=> $4::vector))
                         * retention_score * emotional_weight AS rank
                FROM semantic_memories
                WHERE project_id = $1 AND user_id = $2 AND agent_id = $3
                  AND scope IN ('global', $5)
                  AND memory_tier != 'archived'
                  AND superseded_by IS NULL
                ORDER BY rank DESC
                LIMIT $6
                """,
                project_id, user_id, agent_id, emb, allowed_scope, limit,
            )
            if rows:
                # Reinforce on access — stability grows, retention resets.
                await conn.execute(
                    """
                    UPDATE semantic_memories SET
                      reinforcement_count = reinforcement_count + 1,
                      stability           = stability * (1 + 0.2 * retention_score),
                      last_accessed_at    = NOW(),
                      retention_score     = 1.0
                    WHERE id = ANY($1::uuid[])
                    """,
                    [r["id"] for r in rows],
                )
            if enforce_rls:
                await tx.commit()
        return [
            {"id": str(r["id"]), "content": r["content"],
             "memory_type": r["memory_type"], "rank": float(r["rank"]),
             "memory_tier": r["memory_tier"],
             "reinforcement_count": r["reinforcement_count"]}
            for r in rows
        ]
