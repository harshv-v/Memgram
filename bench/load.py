"""Load harness — the hot path under concurrent users.

    python bench/load.py                       # 20 users x 20 turns
    LOAD_USERS=100 LOAD_TURNS=50 python bench/load.py

Each virtual user loops the REAL production pattern per turn:
  GET /v1/instructions  ->  GET /v1/memory/search  ->  POST /v1/ingest
Reports p50/p95/p99 + RPS per endpoint and overall. Run against a stack with
MEMGRAM_FAKE_LLM=1 to load-test OUR code (not OpenAI's rate limits), or a real
stack to measure the whole system.
"""
import asyncio
import os
import statistics
import sys
import time

import httpx

BASE = os.environ.get("MEMGRAM_API_BASE_URL", "http://localhost:8000")
KEY = os.environ.get("MEMGRAM_API_KEY", "mgram_dev_key")
USERS = int(os.environ.get("LOAD_USERS", "20"))
TURNS = int(os.environ.get("LOAD_TURNS", "20"))
PROJECT = f"load-{int(time.time())}"

LAT: dict[str, list[float]] = {"context(combined)": [], "instructions": [], "search": [], "ingest": []}
ERRORS = 0


async def turn(c: httpx.AsyncClient, uid: str, i: int):
    global ERRORS
    q = {"project_id": PROJECT, "agent_id": "load", "user_id": uid}
    try:
        t0 = time.perf_counter()
        await c.get("/v1/context", params={**q, "query": f"topic {i}"})
        LAT["context(combined)"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        await c.get("/v1/instructions", params={**q, "status": "active"})
        LAT["instructions"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        await c.get("/v1/memory/search", params={**q, "query": f"topic {i}", "limit": 5})
        LAT["search"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        await c.post("/v1/ingest", json={
            **q, "messages": [{"role": "user", "content": f"user {uid} message {i} about topic {i % 7}"}],
            "response_text": "ok"})
        LAT["ingest"].append(time.perf_counter() - t0)
    except Exception:
        ERRORS += 1


async def user(uid: str):
    async with httpx.AsyncClient(
            base_url=BASE, headers={"Authorization": f"Bearer {KEY}"}, timeout=30) as c:
        for i in range(TURNS):
            await turn(c, uid, i)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] * 1000


async def main():
    print(f"load: {USERS} users x {TURNS} turns -> {BASE}  (project={PROJECT})")
    t0 = time.perf_counter()
    await asyncio.gather(*(user(f"u{i}") for i in range(USERS)))
    wall = time.perf_counter() - t0
    total = sum(len(v) for v in LAT.values())
    print(f"\n{'endpoint':<14}{'n':>7}{'p50':>9}{'p95':>9}{'p99':>9}{'mean':>9}")
    print("-" * 57)
    for name, xs in LAT.items():
        if xs:
            print(f"{name:<14}{len(xs):>7}{pct(xs, .5):>8.1f}ms{pct(xs, .95):>8.1f}ms"
                  f"{pct(xs, .99):>8.1f}ms{statistics.mean(xs)*1000:>8.1f}ms")
    print("-" * 57)
    print(f"total {total} requests in {wall:.1f}s = {total/wall:.0f} rps | errors: {ERRORS}")
    sys.exit(1 if ERRORS else 0)


if __name__ == "__main__":
    asyncio.run(main())
