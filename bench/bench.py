"""Memgram latency benchmark — measures the hot-path overhead the SDK adds to a
request, and how semantic search scales with memory count.

It does NOT benchmark the LLM completion itself (that's the provider's latency,
not ours). It measures exactly what Memgram inserts into the hot path:

  - instruction fetch     GET /v1/instructions   (Valkey-cached)
  - semantic search       GET /v1/memory/search  (embedding + pgvector HNSW + reinforce)
  - ingest (post-hook)    POST /v1/ingest        (should return ~instantly, 202)

For search it reports two numbers so you can see where the time goes:
  - cached query : same text every call -> embedding is cached -> pgvector-only cost
  - fresh query  : unique text each call -> full cost (OpenAI embedding + pgvector)
The difference ≈ the embedding round-trip.

Seeds synthetic rows (random unit vectors; relevance irrelevant for latency) and
sweeps the corpus size to show HNSW scaling.

Run (stack up):
    DATABASE_URL=postgresql://memgram:memgram@localhost:5433/memgram \
    OPENAI_API_KEY=sk-... python bench/bench.py
"""
import asyncio
import os
import random
import sys
import time

import asyncpg
import httpx

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

API = os.environ.get("MEMGRAM_API_BASE_URL", "http://localhost:8000")
KEY = os.environ.get("MEMGRAM_API_KEY", "mgram_dev_key")
DSN = os.environ.get("DATABASE_URL", "postgresql://memgram:memgram@localhost:5433/memgram")
DIMS = int(os.environ.get("MEMGRAM_EMBED_DIMS", "1536"))
PROJECT, AGENT, USER = "bench", "bench-agent", "bench"
ITERS = int(os.environ.get("BENCH_ITERS", "40"))
SCALE = [int(x) for x in os.environ.get("BENCH_SCALE", "100,1000,10000").split(",")]
H = {"Authorization": f"Bearer {KEY}"}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def report(name, ms):
    print(f"  {name:<28} p50={pct(ms,50):7.1f}ms  p95={pct(ms,95):7.1f}ms  "
          f"p99={pct(ms,99):7.1f}ms  (n={len(ms)})")


def time_calls(fn, n):
    out = []
    for i in range(n):
        t = time.perf_counter()
        fn(i)
        out.append((time.perf_counter() - t) * 1000)
    return out


async def seed_to(pool, target):
    """Top the bench corpus up to `target` rows with random unit vectors."""
    have = await pool.fetchval(
        "SELECT count(*) FROM semantic_memories WHERE user_id=$1", USER)
    need = target - have
    if need <= 0:
        return
    rows = []
    for i in range(need):
        v = [random.gauss(0, 1) for _ in range(DIMS)]
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        vec = "[" + ",".join(f"{x / norm:.6f}" for x in v) + "]"
        rows.append((PROJECT, AGENT, USER, f"bench memory {have + i}", "fact", "bench", vec))
    await pool.executemany(
        """INSERT INTO semantic_memories
           (project_id, agent_id, user_id, content, memory_type, source, embedding)
           VALUES ($1,$2,$3,$4,$5,$6,$7::vector)""", rows)


def bench_http():
    client = httpx.Client(base_url=API, headers=H, timeout=30)
    instr_params = {"project_id": PROJECT, "agent_id": AGENT, "user_id": USER, "status": "active"}

    def instr(_):
        client.get("/v1/instructions", params=instr_params).raise_for_status()

    def search_cached(_):
        client.get("/v1/memory/search", params={
            "project_id": PROJECT, "agent_id": AGENT, "user_id": USER,
            "query": "what stack does the user prefer", "limit": 5}).raise_for_status()

    def search_fresh(i):
        client.get("/v1/memory/search", params={
            "project_id": PROJECT, "agent_id": AGENT, "user_id": USER,
            "query": f"unique benchmark query {time.time()} {i}", "limit": 5}).raise_for_status()

    def ingest(i):
        client.post("/v1/ingest", json={
            "project_id": PROJECT, "agent_id": AGENT, "user_id": USER,
            "messages": [{"role": "user", "content": f"ping {i}"}],
            "response_text": "pong"}).raise_for_status()

    # warm caches / connections
    for f in (instr, search_cached, search_fresh, ingest):
        f(0)
    print(f"\nHOT-PATH COMPONENTS (corpus held at {SCALE[-1]} rows, {ITERS} iters):")
    report("instructions (cached)", time_calls(instr, ITERS))
    report("search (embedding cached)", time_calls(search_cached, ITERS))
    report("search (fresh embedding)", time_calls(search_fresh, ITERS))
    report("ingest / post-hook (202)", time_calls(ingest, ITERS))
    client.close()


async def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set on the *server* side — fresh-search "
              "numbers reflect the fake embedder, not real OpenAI.")
    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=4)
    client = httpx.Client(base_url=API, headers=H, timeout=30)

    print("=" * 72)
    print("MEMGRAM LATENCY BENCHMARK")
    print("=" * 72)
    print(f"API={API}  dims={DIMS}  corpus sizes={SCALE}")

    # Scaling sweep: grow the corpus, measure search at each size.
    print("\nSEARCH SCALING (pgvector HNSW) — p50 / p95 over", ITERS, "iters:")
    def search_cached(_):
        client.get("/v1/memory/search", params={
            "project_id": PROJECT, "agent_id": AGENT, "user_id": USER,
            "query": "stable benchmark query", "limit": 5}).raise_for_status()
    def search_fresh(i):
        client.get("/v1/memory/search", params={
            "project_id": PROJECT, "agent_id": AGENT, "user_id": USER,
            "query": f"fresh {time.time()} {i}", "limit": 5}).raise_for_status()
    for n in SCALE:
        await seed_to(pool, n)
        search_cached(0); search_fresh(0)
        c = time_calls(search_cached, ITERS)
        f = time_calls(search_fresh, ITERS)
        print(f"  {n:>7,} rows   cached(pgvector): p50={pct(c,50):6.1f} p95={pct(c,95):6.1f}ms"
              f"   fresh(embed+pgvector): p50={pct(f,50):6.1f} p95={pct(f,95):6.1f}ms")
    client.close()
    await pool.close()

    bench_http()

    print("\nNotes:")
    print("  - cached vs fresh search delta ≈ the OpenAI embedding round-trip.")
    print("  - 'ingest' returns 202 before any extraction runs (fire-and-forget).")
    print("  - LLM completion latency is the provider's, not measured here.")
    print("  - run `python bench/bench.py` again to reuse the seeded corpus.\n")


if __name__ == "__main__":
    asyncio.run(main())
