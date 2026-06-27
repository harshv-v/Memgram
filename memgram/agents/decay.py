"""Decay — nightly Ebbinghaus forgetting curve. Pure SQL, zero LLM cost.

R = exp(-t / S): t = days since last access, S = stability.
Tiers: active (R > 0.3) → fading (0.1 < R ≤ 0.3) → archived (R ≤ 0.1).
'promoted' memories are never decayed.

Also flags instructions unconfirmed for `demotion_days` (default 90) for
demotion review (status='review') — the unlearning path.
"""
DECAY_SQL = """
UPDATE semantic_memories SET
  retention_score = EXP(
    -(EXTRACT(EPOCH FROM (NOW() - last_accessed_at)) / 86400.0) / stability
  )
WHERE memory_tier NOT IN ('promoted');
"""

TIER_SQL = """
UPDATE semantic_memories SET
  memory_tier = CASE
    WHEN retention_score >  0.3 THEN 'active'
    WHEN retention_score >  0.1 THEN 'fading'
    ELSE 'archived'
  END
WHERE memory_tier NOT IN ('promoted')
  AND superseded_by IS NULL;  -- never resurrect a superseded (deprecated) fact
"""

DEMOTION_SQL = """
UPDATE instructions SET status = 'review'
WHERE status = 'active'
  AND source = 'agent_proposed'
  AND last_confirmed_at < NOW() - ($1 || ' days')::interval
RETURNING id;
"""


class DecayAgent:
    """Not a BaseAgent — no LLM. Same .run(job) contract for the dispatcher."""

    def __init__(self, store, config: dict | None = None):
        self.store = store
        self.config = config or {}

    async def run(self, job: dict) -> dict:
        demotion_days = str(int(self.config.get("demotion_days", 90)))
        async with self.store.pool.acquire() as conn:
            decayed = await conn.execute(DECAY_SQL)
            tiered = await conn.execute(TIER_SQL)
            flagged = await conn.fetch(DEMOTION_SQL, demotion_days)
        return {
            "decayed": decayed, "retiered": tiered,
            "instructions_flagged_for_review": len(flagged),
        }
