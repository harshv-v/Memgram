"""The gauntlet — one user's memory life, three simulated weeks, asserted.

Where test_pipeline checks each mechanism in isolation, the gauntlet checks
the STORY: mechanisms composing in sequence on one user, the way production
actually exercises them.

  Week 1: introductions -> extraction; repetition -> reinforcement not duplication;
          a correction -> double emotional weight.
  Week 2: the user moves cities (contradiction v2) -> old fact superseded,
          retrieval returns the NEW city only.
  Week 3: habit threshold -> reflection -> proposal (pending) -> human approves
          -> instruction active. Time machine ages 45 days -> decay archives the
          unused fact, but the approved instruction survives (never decays).
  Always: a second agent can't see the first agent's private memories.

    DATABASE_URL=... MEMGRAM_FAKE_LLM=1 MEMGRAM_FAKE_REDIS=1 python tests/test_gauntlet.py
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

PROJECT, AGENT, USER = "gauntlet", "assistant", "g-user"
checks = []
def ck(n, c): checks.append((n, bool(c)))


def ids(agent=AGENT):
    return {"project_id": PROJECT, "agent_id": agent, "user_id": USER}


def convo(text):
    return {"messages": [{"role": "user", "content": text}], "response_text": "Noted."}


async def mem_rows(pool, agent=AGENT, tier=None):
    q = """SELECT content, memory_tier, reinforcement_count, emotional_weight,
                  superseded_by FROM semantic_memories
           WHERE project_id=$1 AND user_id=$2 AND agent_id=$3"""
    args = [PROJECT, USER, agent]
    if tier:
        q += " AND memory_tier = $4"
        args.append(tier)
    async with pool.acquire() as c:
        return [dict(r) for r in await c.fetch(q, *args)]


async def main():
    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=1, max_size=4)
    embedder = get_embedder()
    store = MemoryStore(pool, embedder)
    retriever = Retriever(pool, embedder)
    queue = JobQueue()
    disp = Dispatcher(store, queue, embedder,
                      config={"reflect_every_n": 9999,
                              "reflection": {"min_logs": 1, "habit_threshold": 5}})
    async with pool.acquire() as c:
        for t in ("instructions", "semantic_memories", "episodic_logs"):
            await c.execute(f"DELETE FROM {t} WHERE project_id=$1", PROJECT)

    # ---- WEEK 1: introductions --------------------------------------------------
    queue.enqueue("extract", {**ids(), **convo("Hey, I work in Rust and live in Berlin.")})
    await disp.drain()
    rows = await mem_rows(pool)
    all_text = " ".join(r["content"].lower() for r in rows)
    ck("wk1: introductions extracted (rust + berlin)",
       "rust" in all_text and "berlin" in all_text)

    # repetition across days -> reinforcement, not duplication
    for day in range(3):
        queue.enqueue("extract", {**ids(), **convo(f"Day {day}: more Rust work today.")})
    await disp.drain()
    rows = await mem_rows(pool)
    rust = next(r for r in rows if "rust" in r["content"].lower())
    ck("wk1: repetition reinforced one row (no duplicates)",
       rust["reinforcement_count"] >= 3
       and sum("rust" in r["content"].lower() for r in rows) == 1)

    # a correction lands with double weight
    queue.enqueue("extract", {**ids(), **convo("No, that's wrong - I asked for brief answers.")})
    await disp.drain()
    rows = await mem_rows(pool)
    corr = [r for r in rows if r["emotional_weight"] >= 2.0]
    ck("wk1: correction stored with emotional_weight 2.0", len(corr) >= 1)

    # ---- WEEK 2: the user moves (contradiction v2, per-job flag) -----------------
    queue.enqueue("extract", {**ids(), **convo("Big news, I moved to Munich!"),
                              "contradiction": True})
    await disp.drain()
    rows = await mem_rows(pool)
    berlin = next((r for r in rows if "berlin" in r["content"].lower()), None)
    munich = next((r for r in rows if "munich" in r["content"].lower()), None)
    ck("wk2: new city stored active", munich and munich["memory_tier"] == "active")
    ck("wk2: old city superseded + archived",
       berlin and berlin["memory_tier"] == "archived" and berlin["superseded_by"])
    hits = await retriever.search(PROJECT, AGENT, USER, "Where does the user live?", limit=5)
    hit_text = " ".join(h["content"].lower() for h in hits)
    ck("wk2: retrieval returns Munich, never Berlin",
       "munich" in hit_text and "berlin" not in hit_text)

    # ---- WEEK 3: habit -> proposal -> human approval -----------------------------
    for i in range(5):
        queue.enqueue("extract", {**ids(), **convo(f"Rust question #{i}: lifetimes?")})
    await disp.drain()
    await store.log_episodic(role="user", content="another rust question", **ids())
    queue.enqueue("reflect", ids())
    await disp.drain()   # reflect -> enqueues propose
    await disp.drain()   # propose -> pending instruction
    async with pool.acquire() as c:
        prop = await c.fetchrow(
            "SELECT id, status FROM instructions WHERE project_id=$1 AND user_id=$2 "
            "AND source='agent_proposed'", PROJECT, USER)
    ck("wk3: habit produced a PENDING proposal (trust gate)",
       prop and prop["status"] == "pending")

    # the human approves -> active instruction
    async with pool.acquire() as c:
        await c.execute("UPDATE instructions SET status='active' WHERE id=$1", prop["id"])
        active = await c.fetchval(
            "SELECT COUNT(*) FROM instructions WHERE project_id=$1 AND user_id=$2 "
            "AND status='active'", PROJECT, USER)
    ck("wk3: approved proposal is now an active instruction", active == 1)

    # ---- TIME MACHINE: 45 days pass ----------------------------------------------
    async with pool.acquire() as c:
        await c.execute(
            "UPDATE semantic_memories SET last_accessed_at = NOW() - INTERVAL '45 days' "
            "WHERE project_id=$1 AND user_id=$2", PROJECT, USER)
    queue.enqueue("decay", {})
    await disp.drain()
    rows = await mem_rows(pool)
    corr45 = next(r for r in rows if r["emotional_weight"] >= 2.0)
    weak = [r for r in rows if r["memory_tier"] == "archived"]
    strong = [r for r in rows if r["memory_tier"] in ("active", "fading", "promoted")]
    ck("45d: weakly-reinforced memories decayed to archived", len(weak) >= 2)
    ck("45d: heavily-reinforced memories survive decay",
       any(r["reinforcement_count"] >= 4 for r in strong))
    _ = corr45  # weight persists on the row regardless of tier
    async with pool.acquire() as c:
        instr = await c.fetchval(
            "SELECT COUNT(*) FROM instructions WHERE project_id=$1 AND user_id=$2 "
            "AND status='active'", PROJECT, USER)
    ck("45d: the approved instruction NEVER decays", instr == 1)

    # ---- scope: a second agent sees nothing private -------------------------------
    other = await retriever.search(PROJECT, "billing-bot", USER, "rust?", limit=5)
    ck("scope: another agent can't read this agent's private memories",
       len(other) == 0)

    async with pool.acquire() as c:
        for t in ("instructions", "semantic_memories", "episodic_logs"):
            await c.execute(f"DELETE FROM {t} WHERE project_id=$1", PROJECT)
    await pool.close()

    fail = False
    for n, c in checks:
        print(f"{'PASS' if c else 'FAIL'}  {n}")
        fail = fail or not c
    print(f"\n{sum(c for _, c in checks)}/{len(checks)} gauntlet checks passed")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    asyncio.run(main())
