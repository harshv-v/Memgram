"""Reliability suite: transactional outbox + Streams crash recovery.

Proves the ingest dual-write gap is closed:
  1. Turn + job intents commit atomically (queue down -> no data loss).
  2. The relay re-dispatches stranded rows.
  3. Re-dispatch after a mark-crash does NOT duplicate the job (idempotency).
  4. A worker that dies mid-job has its job reclaimed by another consumer.

Run:  DATABASE_URL=... MEMGRAM_FAKE_REDIS=1 python tests/test_outbox.py
"""
import asyncio
import os
import sys
import types

os.environ.setdefault("MEMGRAM_FAKE_REDIS", "1")
os.environ.setdefault("MEMGRAM_FAKE_LLM", "1")

import asyncpg  # noqa: E402

from memgram.api.routes.ingest import IngestBody, ingest  # noqa: E402
from memgram.memory.embedder import get_embedder  # noqa: E402
from memgram.memory.store import MemoryStore  # noqa: E402
from memgram.worker import outbox  # noqa: E402
from memgram.worker.queue import JobQueue  # noqa: E402

checks = []
def ck(n, c): checks.append((n, bool(c)))

PROJECT, AGENT, USER = "outbox-test", "agent", "u-outbox"


class BrokenQueue(JobQueue):
    """Queue whose enqueue always fails — simulates Valkey being down."""
    def enqueue(self, *a, **kw):
        raise ConnectionError("queue down")


def fake_request(pool, store, queue):
    app = types.SimpleNamespace(state=types.SimpleNamespace(
        pool=pool, store=store, queue=queue))
    return types.SimpleNamespace(app=app)


async def main():
    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=1, max_size=4)
    store = MemoryStore(pool, get_embedder())
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM outbox")
        await conn.execute("DELETE FROM episodic_logs WHERE project_id = $1", PROJECT)

    body = IngestBody(project_id=PROJECT, agent_id=AGENT, user_id=USER,
                      messages=[{"role": "user", "content": "I work in Rust."}],
                      response_text="Noted!")

    # --- 1) queue down: turn + intent survive atomically --------------------
    try:
        await ingest(fake_request(pool, store, BrokenQueue()), body)
    except ConnectionError:
        pass  # fast-path dispatch failed — expected
    async with pool.acquire() as conn:
        pending = await conn.fetchval(
            "SELECT COUNT(*) FROM outbox WHERE dispatched_at IS NULL")
        episodic = await conn.fetchval(
            "SELECT COUNT(*) FROM episodic_logs WHERE project_id = $1", PROJECT)
    ck("queue down: outbox intent persisted", pending == 1)
    ck("queue down: episodic rows persisted (atomic with intent)", episodic == 2)

    # --- 2) relay rescues the stranded row ----------------------------------
    q = JobQueue()
    n = await outbox.dispatch(pool, q, older_than_s=0)
    ck("relay dispatched the stranded job", n == 1 and q.depth() == 1)
    async with pool.acquire() as conn:
        pending = await conn.fetchval(
            "SELECT COUNT(*) FROM outbox WHERE dispatched_at IS NULL")
    ck("relay marked the row dispatched", pending == 0)

    # --- 3) enqueue-then-crash-before-mark: no duplicate --------------------
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE outbox SET dispatched_at = NULL")  # simulate mark never landing
    n = await outbox.dispatch(pool, q, older_than_s=0)
    ck("re-dispatch after mark-crash: idempotency key blocks duplicate",
       q.depth() == 1)  # still exactly one live job

    # --- 4) Streams crash recovery: dead consumer's job is reclaimed --------
    w1 = JobQueue(consumer="worker-1")
    job = w1.dequeue(timeout=1)
    ck("worker-1 took the job", job is not None and job["type"] == "extract")
    # worker-1 "crashes" (no ack). worker-2 reclaims after idle threshold 0ms.
    w2 = JobQueue(consumer="worker-2")
    reclaimed = w2.reclaim(min_idle_ms=0)
    ck("worker-2 reclaimed the crashed worker's job",
       any(j["type"] == "extract" for j in reclaimed))
    if reclaimed:
        w2.ack(reclaimed[0])
    ck("after ack: stream fully drained", q.depth() == 0)

    await pool.close()
    fail = False
    for n_, c in checks:
        print(f"{'PASS' if c else 'FAIL'}  {n_}")
        fail = fail or not c
    print(f"\n{sum(c for _, c in checks)}/{len(checks)} outbox/reliability checks passed")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    asyncio.run(main())
