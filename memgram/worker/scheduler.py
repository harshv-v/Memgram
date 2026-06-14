"""Scheduler — cron without cron. A 60s loop that enqueues:
  - decay: nightly after 02:00 UTC, once per day (idempotency-keyed by date)
  - reflect sweep: every hour, one reflect job per (project,user,agent) with
    unreflected logs older than `reflect_every_hrs` (default 24h)
Idempotency keys make this safe to run on multiple workers.
"""
import asyncio
import datetime
import logging

logger = logging.getLogger("memgram.worker")

SWEEP_SQL = """
SELECT DISTINCT project_id, user_id, agent_id FROM episodic_logs
WHERE NOT reflected AND created_at < NOW() - ($1 || ' hours')::interval
"""


class Scheduler:
    def __init__(self, pool, queue, config: dict | None = None):
        self.pool = pool
        self.queue = queue
        self.config = config or {}

    async def tick(self) -> None:
        now = datetime.datetime.now(datetime.timezone.utc)
        if now.hour >= 2:  # nightly decay, once per UTC day
            self.queue.enqueue("decay", {}, idempotency_key=f"decay:{now.date()}")
        if now.minute == 0:  # hourly reflect sweep
            hrs = str(int(self.config.get("reflect_every_hrs", 24)))
            async with self.pool.acquire() as conn:
                groups = await conn.fetch(SWEEP_SQL, hrs)
            for g in groups:
                self.queue.enqueue("reflect", dict(g),
                    idempotency_key=f"reflect:{g['project_id']}:{g['user_id']}:"
                                    f"{g['agent_id']}:{now:%Y-%m-%d-%H}")

    async def run_forever(self) -> None:
        logger.info("memgram scheduler started")
        while True:
            try:
                await self.tick()
            except Exception:
                logger.exception("scheduler tick failed")
            await asyncio.sleep(60)
