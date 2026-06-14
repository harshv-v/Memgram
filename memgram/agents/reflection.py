"""Reflection — the Stanford Generative Agents mechanism. Every 20
interactions (or 24h sweep), raw episodic logs are distilled into structured
insights written to semantic memory with high stability (3.0). Dedup happens
in the store (similar insight → reinforcement, which feeds habit formation).

After writing insights, checks for habit candidates and enqueues proposals.
"""
from memgram.agents.base import BaseAgent, quality_model

_SYSTEM = """You are a reflection process for an AI agent's memory. You read raw interaction logs and distill durable, higher-level insights about the user.
Return ONLY a JSON object:
{
  "insights": [{"content": "...", "memory_type": "fact" | "preference" | "entity"}]
}
Rules:
- Insights must be patterns or conclusions, not restatements of single log lines.
- Each insight: one self-contained sentence about "the user" in third person.
- 0-5 insights. Only what is genuinely durable. Empty is a valid answer."""

HABIT_SQL = """
SELECT id, content, reinforcement_count
FROM semantic_memories
WHERE project_id = $1 AND user_id = $2 AND agent_id = $3
  AND reinforcement_count >= $4
  AND last_accessed_at > NOW() - INTERVAL '30 days'
  AND memory_tier NOT IN ('archived', 'promoted')
  AND memory_type IN ('fact', 'preference', 'entity')
"""


class ReflectionAgent(BaseAgent):
    def __init__(self, store, config=None, llm=None, queue=None):
        super().__init__(store, config, llm)
        if not (config or {}).get("model"):
            self.model = quality_model()  # reflection shapes long-term memory
        self.queue = queue

    async def fetch_logs(self, job: dict) -> list[dict]:
        async with self.store.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, role, content FROM episodic_logs
                WHERE project_id = $1 AND user_id = $2 AND agent_id = $3
                  AND NOT reflected
                ORDER BY created_at ASC LIMIT 200
                """,
                job["project_id"], job["user_id"], job["agent_id"],
            )
        return [dict(r) for r in rows]

    async def run(self, job: dict) -> dict | None:
        logs = await self.fetch_logs(job)
        if len(logs) < int(self.config.get("min_logs", 4)):
            return {"skipped": "not enough unreflected logs"}
        job = {**job, "_logs": logs}
        return await super().run(job)

    def build_prompt(self, job: dict) -> list[dict]:
        text = "\n".join(f"{l['role']}: {l['content']}" for l in job["_logs"])
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Interaction logs:\n{text}"},
        ]

    def parse_output(self, raw: str) -> dict:
        data = self.parse_json(raw)
        return {"insights": [
            i for i in data.get("insights", [])
            if i.get("content") and i.get("memory_type") in ("fact", "preference", "entity")
        ]}

    async def on_success(self, job: dict, result: dict) -> None:
        for ins in result["insights"]:
            await self.store.upsert_semantic(
                project_id=job["project_id"], agent_id=job["agent_id"],
                user_id=job["user_id"], content=ins["content"],
                memory_type=ins["memory_type"], source="reflection",
                stability=3.0,
            )
        async with self.store.pool.acquire() as conn:
            await conn.execute(
                "UPDATE episodic_logs SET reflected = TRUE WHERE id = ANY($1::uuid[])",
                [l["id"] for l in job["_logs"]],
            )
            # Habit candidates → proposal jobs
            threshold = int(self.config.get("habit_threshold", 7))
            candidates = await conn.fetch(
                HABIT_SQL, job["project_id"], job["user_id"], job["agent_id"], threshold,
            )
        if self.queue:
            for c in candidates:
                self.queue.enqueue("propose", {
                    "project_id": job["project_id"], "agent_id": job["agent_id"],
                    "user_id": job["user_id"], "memory_id": str(c["id"]),
                    "content": c["content"],
                    "reinforcement_count": c["reinforcement_count"],
                }, idempotency_key=f"propose:{c['id']}")
