"""Head-to-head: Memgram vs Mem0, same OpenAI models, same conversations.

Fairness notes (read these before trusting any number):
  - Both use gpt-4o-mini for extraction and text-embedding-3-small for embeddings.
  - Storage backends differ by design: Mem0 -> local Qdrant; Memgram -> pgvector
    over the docker API (a localhost network hop). This is a real difference, not
    noise — we're comparing *systems*, not just algorithms.
  - WRITE PATH IS ASYMMETRIC and we report it honestly:
      * Mem0 .add() is synchronous — it runs the extraction LLM call inline, so
        it's slow but the memory is queryable the moment it returns.
      * Memgram ingest is async — it returns an ack in ~ms and a background worker
        extracts later. We report BOTH the ack latency AND time-to-queryable.
    Comparing Mem0's add to Memgram's ack alone would be dishonest; don't.
  - Single client, local machine. Not a distributed/throughput benchmark.

Run (Memgram stack up via docker compose; `pip install mem0ai` first):
    OPENAI_API_KEY=sk-... python bench/compare_mem0.py
"""
import os
import sys
import time
import uuid
import warnings

import httpx

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

API = os.environ.get("MEMGRAM_API_BASE_URL", "http://localhost:8000")
KEY = os.environ.get("MEMGRAM_API_KEY", "mgram_dev_key")
H = {"Authorization": f"Bearer {KEY}"}
MODEL_LLM = "gpt-4o-mini"
MODEL_EMB = "text-embedding-3-small"
SEARCH_ITERS = int(os.environ.get("CMP_SEARCH_ITERS", "10"))

# Labeled conversations: each user message carries a fact we expect to be
# extracted; `kw` are the keywords that must appear in *some* stored memory.
DATA = [
    ("I work in Rust and prefer concise answers.", ["rust"]),
    ("I'm building a REST API with FastAPI in Python.", ["fastapi"]),
    ("I'm allergic to peanuts.", ["peanut"]),
    ("I live in Berlin.", ["berlin"]),
    ("My favorite database is PostgreSQL.", ["postgres"]),
    ("I usually deploy on Fridays.", ["friday"]),
    ("I drive a Tesla Model 3.", ["tesla"]),
    ("I'm vegetarian.", ["vegetarian"]),
]
QUERY = "what do you know about me?"


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def recall(memories_text, data):
    """Fraction of expected facts whose keyword appears in any stored memory."""
    blob = " ".join(memories_text).lower()
    hit = sum(all(k in blob for k in kw) for _, kw in data)
    return hit, len(data)


# ---------------------------------------------------------------- Mem0 --------
def run_mem0():
    from mem0 import Memory
    path = f"/tmp/qdrant_cmp_{uuid.uuid4().hex[:8]}"
    cfg = {
        "llm": {"provider": "openai", "config": {"model": MODEL_LLM}},
        "embedder": {"provider": "openai", "config": {"model": MODEL_EMB}},
        "vector_store": {"provider": "qdrant", "config": {"path": path, "on_disk": True}},
    }
    m = Memory.from_config(cfg)
    user = "cmp_user"
    # warm up (first add creates the collection — don't count it)
    m.add([{"role": "user", "content": "warmup message"}], user_id="warm")

    add_ms = []
    for text, _ in DATA:
        conv = [{"role": "user", "content": text}, {"role": "assistant", "content": "Noted."}]
        t = time.perf_counter()
        m.add(conv, user_id=user)
        add_ms.append((time.perf_counter() - t) * 1000)

    # memory is queryable immediately after add returns (sync)
    got = m.get_all(filters={"user_id": user}, top_k=100)
    mems = [r["memory"] for r in (got.get("results") if isinstance(got, dict) else got)]
    hit, total = recall(mems, DATA)

    search_ms = []
    for _ in range(SEARCH_ITERS):
        t = time.perf_counter()
        m.search(QUERY, filters={"user_id": user}, top_k=5)
        search_ms.append((time.perf_counter() - t) * 1000)

    return {"add": add_ms, "search": search_ms, "recall": (hit, total),
            "n_mem": len(mems), "queryable_lag_ms": 0.0}


# -------------------------------------------------------------- Memgram ------
def run_memgram():
    client = httpx.Client(base_url=API, headers=H, timeout=60)
    proj, agent, user = f"cmp-{uuid.uuid4().hex[:6]}", "cmp-agent", "cmp_user"

    def list_mems():
        r = client.get("/v1/memory", params={
            "project_id": proj, "agent_id": agent, "user_id": user, "limit": 100})
        r.raise_for_status()
        return [m["content"] for m in r.json()["memories"]]

    # warm connection
    client.get("/v1/instructions", params={
        "project_id": proj, "agent_id": agent, "user_id": user, "status": "active"})

    ack_ms = []
    t_first = time.perf_counter()
    for text, _ in DATA:
        body = {"project_id": proj, "agent_id": agent, "user_id": user,
                "messages": [{"role": "user", "content": text}], "response_text": "Noted."}
        t = time.perf_counter()
        client.post("/v1/ingest", json=body).raise_for_status()
        ack_ms.append((time.perf_counter() - t) * 1000)

    # time-to-queryable: poll until the worker has extracted ~all facts (or timeout)
    deadline = time.perf_counter() + 60
    hit = 0
    while time.perf_counter() < deadline:
        mems = list_mems()
        hit, total = recall(mems, DATA)
        if hit >= total:
            break
        time.sleep(0.5)
    queryable_lag = (time.perf_counter() - t_first) * 1000

    mems = list_mems()
    hit, total = recall(mems, DATA)

    search_ms = []
    for _ in range(SEARCH_ITERS):
        t = time.perf_counter()
        client.get("/v1/memory/search", params={
            "project_id": proj, "agent_id": agent, "user_id": user,
            "query": QUERY, "limit": 5}).raise_for_status()
        search_ms.append((time.perf_counter() - t) * 1000)

    client.close()
    return {"ack": ack_ms, "search": search_ms, "recall": (hit, total),
            "n_mem": len(mems), "queryable_lag_ms": queryable_lag}


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (both systems use it for LLM + embeddings).")
    print("=" * 70)
    print(f"MEMGRAM vs MEM0  |  LLM={MODEL_LLM}  emb={MODEL_EMB}  facts={len(DATA)}")
    print("=" * 70)

    print("\nRunning Mem0 (sync add — this is slow by design)...")
    mz = run_mem0()
    print("Running Memgram (async ingest + background worker)...")
    mg = run_memgram()

    print("\n--- WRITE PATH ----------------------------------------------------")
    print(f"  Mem0    add (sync, incl. extraction): p50={pct(mz['add'],50):7.0f}ms "
          f"p95={pct(mz['add'],95):7.0f}ms  -> queryable immediately")
    print(f"  Memgram ingest ack (async):           p50={pct(mg['ack'],50):7.0f}ms "
          f"p95={pct(mg['ack'],95):7.0f}ms  -> queryable after ~{mg['queryable_lag_ms']/1000:.1f}s (worker)")

    print("\n--- SEARCH LATENCY ------------------------------------------------")
    print(f"  Mem0    search: p50={pct(mz['search'],50):7.1f}ms  p95={pct(mz['search'],95):7.1f}ms")
    print(f"  Memgram search: p50={pct(mg['search'],50):7.1f}ms  p95={pct(mg['search'],95):7.1f}ms")

    print("\n--- EXTRACTION RECALL (facts captured / total) --------------------")
    print(f"  Mem0   : {mz['recall'][0]}/{mz['recall'][1]}  ({mz['n_mem']} memories stored)")
    print(f"  Memgram: {mg['recall'][0]}/{mg['recall'][1]}  ({mg['n_mem']} memories stored)")

    print("\nCaveats: different vector backends (Qdrant local vs pgvector-over-docker);")
    print("write paths are sync vs async (hence the asymmetric reporting); single client.\n")


if __name__ == "__main__":
    main()
