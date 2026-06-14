"""Background-pipeline verification — real Postgres, fake LLM, fake Redis.

Exercises the parts the thin slice doesn't: the full cognitive loop end to end.
  1. extract: a conversation becomes semantic memories.
  2. retrieve: ranked search returns them and reinforces on access.
  3. dedup/reinforce: re-extracting the same fact bumps reinforcement_count, not row count.
  4. reflect -> propose: a reinforced pattern becomes a PENDING instruction (trust gate).
  5. decay: an untouched memory's retention falls and it re-tiers (Ebbinghaus).
  6. summarize: a long conversation compresses to one episodic row.

Run:
    DATABASE_URL=postgresql://... MEMGRAM_FAKE_LLM=1 MEMGRAM_FAKE_REDIS=1 \
        python tests/test_pipeline.py
"""
import asyncio
import os
import sys

import asyncpg

os.environ.setdefault("MEMGRAM_FAKE_LLM", "1")
os.environ.setdefault("MEMGRAM_FAKE_REDIS", "1")

from memgram.memory.embedder import get_embedder
from memgram.memory.retriever import Retriever
from memgram.memory.store import MemoryStore
from memgram.worker.dispatcher import Dispatcher
from memgram.worker.queue import JobQueue

PROJECT = "pipe-project"
AGENT = "pipe-agent"
RUST_CONVO = [
    {"role": "user", "content": "I work in Rust and I prefer concise answers."},
    {"role": "assistant", "content": "Got it."},
]
checks: list[tuple[str, bool]] = []


def check(name, ok):
    checks.append((name, bool(ok)))


async def clean(pool, user):
    async with pool.acquire() as c:
        for t in ("instructions", "semantic_memories", "episodic_logs"):
            await c.execute(f"DELETE FROM {t} WHERE project_id=$1 AND user_id=$2", PROJECT, user)


def ids(user):
    return {"project_id": PROJECT, "agent_id": AGENT, "user_id": user}


async def main():
    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=1, max_size=4)
    embedder = get_embedder()
    store = MemoryStore(pool, embedder)
    retriever = Retriever(pool, embedder)
    queue = JobQueue()
    disp = Dispatcher(store, queue, embedder,
                      config={"reflect_every_n": 9999,
                              "reflection": {"min_logs": 1, "habit_threshold": 7}})

    # -- 1. extract --------------------------------------------------------
    u = "pipe-extract"
    await clean(pool, u)
    queue.enqueue("extract", {**ids(u), "messages": RUST_CONVO, "response_text": "Got it."})
    await disp.drain()
    async with pool.acquire() as c:
        rows = await c.fetch(
            "SELECT content FROM semantic_memories WHERE project_id=$1 AND user_id=$2",
            PROJECT, u)
    contents = " ".join(r["content"].lower() for r in rows)
    check("extract wrote semantic memories", len(rows) >= 1)
    check("extract captured the Rust preference", "rust" in contents)

    # -- 2. retrieve + reinforce ------------------------------------------
    hits = await retriever.search(PROJECT, AGENT, u, "what language does the user use?", limit=5)
    check("retriever returns memories", len(hits) >= 1)
    async with pool.acquire() as c:
        rc = await c.fetchval(
            "SELECT MAX(reinforcement_count) FROM semantic_memories WHERE project_id=$1 AND user_id=$2",
            PROJECT, u)
    check("access reinforces (count > 1)", rc and rc > 1)

    # -- 3. dedup: re-extract same convo, row count stays, count rises -----
    before = len(rows)
    queue.enqueue("extract", {**ids(u), "messages": RUST_CONVO, "response_text": "Got it."})
    await disp.drain()
    async with pool.acquire() as c:
        after = await c.fetchval(
            "SELECT COUNT(*) FROM semantic_memories WHERE project_id=$1 AND user_id=$2",
            PROJECT, u)
    check("dedup keeps row count stable", after == before)

    # -- 4. reflect -> propose (the differentiator + trust gate) -----------
    h = "pipe-habit"
    await clean(pool, h)
    for _ in range(7):  # push a memory over the habit threshold
        await store.upsert_semantic(content="The user works in Rust.", memory_type="preference", **ids(h))
    async with pool.acquire() as c:
        for i in range(4):
            await store.log_episodic(role="user", content=f"rust question {i}", **ids(h))
    queue.enqueue("reflect", ids(h))
    await disp.drain()  # reflect runs, enqueues propose
    await disp.drain()  # propose runs
    async with pool.acquire() as c:
        prop = await c.fetchrow(
            """SELECT status, source FROM instructions
               WHERE project_id=$1 AND user_id=$2 AND source='agent_proposed'""",
            PROJECT, h)
    check("habit pattern produced a proposal", prop is not None)
    check("proposal is PENDING (agent can't self-promote)",
          prop is not None and prop["status"] == "pending")

    # -- 5. decay (pure SQL, Ebbinghaus) -----------------------------------
    d = "pipe-decay"
    await clean(pool, d)
    await store.upsert_semantic(content="A stale fact nobody reuses.", **ids(d))
    async with pool.acquire() as c:
        await c.execute(
            """UPDATE semantic_memories SET last_accessed_at = NOW() - INTERVAL '30 days',
               stability = 1.0 WHERE project_id=$1 AND user_id=$2""", PROJECT, d)
    queue.enqueue("decay", {})
    await disp.drain()
    async with pool.acquire() as c:
        row = await c.fetchrow(
            "SELECT retention_score, memory_tier FROM semantic_memories WHERE project_id=$1 AND user_id=$2",
            PROJECT, d)
    check("decay lowered retention", row and row["retention_score"] < 0.1)
    check("decay archived the stale memory", row and row["memory_tier"] == "archived")

    # -- 6. summarize ------------------------------------------------------
    s = "pipe-sum"
    await clean(pool, s)
    queue.enqueue("summarize", {**ids(s),
                                "messages": RUST_CONVO + [{"role": "user", "content": "more rust"}]})
    await disp.drain()
    async with pool.acquire() as c:
        comp = await c.fetchval(
            "SELECT content FROM episodic_logs WHERE project_id=$1 AND user_id=$2 AND role='compressed_session'",
            PROJECT, s)
    check("summarizer wrote a compressed session", comp is not None and "summary" in comp.lower())

    # cleanup + report
    for u in ("pipe-extract", "pipe-habit", "pipe-decay", "pipe-sum"):
        await clean(pool, u)
    await pool.close()

    failed = False
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failed = failed or not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
