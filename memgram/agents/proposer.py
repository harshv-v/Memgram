"""Proposer — turns a habit candidate into a natural-language instruction
proposal. THE TRUST GATE LIVES HERE TOO: proposals are written with
source='agent_proposed', which the API layer forces to status='pending'.
Only a user can promote them to active.

Skips proposing if a semantically similar active instruction already exists.
Fires the on_proposal webhook if configured (MEMGRAM_WEBHOOK_ON_PROPOSAL).
"""
import logging
import math
import os

import httpx

from memgram.agents.base import BaseAgent, quality_model

logger = logging.getLogger("memgram.agents")

_SYSTEM = """You convert an observed behavioural pattern into a clear, short, imperative instruction an AI assistant should follow for this user.
Return ONLY JSON: {"instruction": "...", "priority": 1 | 2 | 3}
priority: 1 = always applies, 2 = strong preference, 3 = soft preference.
The instruction must be one sentence, actionable, and general (not tied to one conversation)."""


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class ProposerAgent(BaseAgent):
    def __init__(self, store, config=None, llm=None, embedder=None):
        super().__init__(store, config, llm)
        if not (config or {}).get("model"):
            self.model = quality_model()  # proposals are user-facing
        self.embedder = embedder

    async def run(self, job: dict) -> dict | None:
        if await self._similar_instruction_exists(job):
            return {"skipped": "similar instruction already exists"}
        return await super().run(job)

    async def _similar_instruction_exists(self, job: dict) -> bool:
        async with self.store.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT content FROM instructions
                WHERE project_id = $1 AND agent_id = $2 AND user_id = $3
                  AND status IN ('active', 'pending')
                """,
                job["project_id"], job["agent_id"], job["user_id"],
            )
        if not rows or self.embedder is None:
            return False
        cand = await self.embedder.embed(job["content"])
        for r in rows:  # instruction stores are tiny; in-process cosine is fine
            if _cos(cand, await self.embedder.embed(r["content"])) > 0.85:
                return True
        return False

    def build_prompt(self, job: dict) -> list[dict]:
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content":
                f"Pattern observed {job['reinforcement_count']} times: {job['content']}"},
        ]

    def parse_output(self, raw: str) -> dict:
        data = self.parse_json(raw)
        if not data.get("instruction"):
            raise ValueError("no instruction in output")
        return {"instruction": data["instruction"],
                "priority": int(data.get("priority", 2))}

    async def on_success(self, job: dict, result: dict) -> None:
        async with self.store.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO instructions
                  (project_id, agent_id, user_id, content, priority, source, status)
                VALUES ($1,$2,$3,$4,$5,'agent_proposed','pending')
                RETURNING id
                """,
                job["project_id"], job["agent_id"], job["user_id"],
                result["instruction"], result["priority"],
            )
            await conn.execute(
                "UPDATE semantic_memories SET memory_tier = 'promoted' WHERE id = $1::uuid",
                job["memory_id"],
            )
        await self._fire_webhook(job, result, str(row["id"]))

    async def _fire_webhook(self, job, result, proposal_id: str) -> None:
        url = os.environ.get("MEMGRAM_WEBHOOK_ON_PROPOSAL")
        if not url:
            return
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                await c.post(url, json={
                    "event": "proposal.created", "proposal_id": proposal_id,
                    "user_id": job["user_id"], "agent_id": job["agent_id"],
                    "instruction": result["instruction"],
                    "evidence_count": job["reinforcement_count"],
                })
        except Exception as e:
            logger.warning("on_proposal webhook failed: %s", e)
