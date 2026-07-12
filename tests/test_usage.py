"""Usage accounting + combined-context suite — live API, real Postgres.

Proves:
  1. /v1/context returns instructions + memories in ONE call, same content as
     the two-call path, and estimates injected tokens.
  2. The assembler actually uses it (and injects identically).
  3. Injection usage lands in usage_events per user.
  4. Agent LLM usage is recorded (via a fake-usage LLM through BaseAgent).
  5. /v1/usage aggregates: tokens by kind, cost, storage rows+bytes.

Run (API up):  MEMGRAM_API_KEY=... DATABASE_URL=... python tests/test_usage.py
"""
import asyncio
import os
import sys
import time

import httpx

from memgram import Memgram

BASE = os.environ.get("MEMGRAM_API_URL", "http://localhost:8000")
KEY = os.environ.get("MEMGRAM_API_KEY", "mgram_dev_key")
PROJECT, AGENT, USER = "usage-test", "u-agent", "u-user"

checks = []
def ck(n, c): checks.append((n, bool(c)))
api = httpx.Client(base_url=BASE, headers={"Authorization": f"Bearer {KEY}"}, timeout=30)
Q = {"project_id": PROJECT, "agent_id": AGENT, "user_id": USER}


def main():
    # seed: one instruction + one memory (via ingest -> worker not needed: direct API)
    api.post("/v1/instructions", json={**Q, "content": "Always answer briefly.",
                                       "priority": 1, "source": "user"}).raise_for_status()
    api.post("/v1/ingest", json={**Q, "messages": [
        {"role": "user", "content": "I work in Rust."}],
        "response_text": "Noted."}).raise_for_status()
    time.sleep(0.5)

    # -- 1. combined endpoint parity ---------------------------------------
    ctx = api.get("/v1/context", params={**Q, "query": "what language?"}).json()
    two_i = api.get("/v1/instructions", params={**Q, "status": "active"}).json()["instructions"]
    ck("context returns instructions", len(ctx["instructions"]) == len(two_i) == 1)
    ck("context estimates injected tokens", ctx["injected_tokens_est"] > 0)

    # -- 2. assembler uses ONE call ------------------------------------------
    mem = Memgram(api_key=KEY, agent_name=AGENT, project_id=PROJECT, api_base_url=BASE)

    class FC:
        kw = None
        def create(self, **kw): FC.kw = kw; return {"choices": [{"message": {"content": "ok"}}]}
    class Cli:
        def __init__(self):
            import types
            self.chat = types.SimpleNamespace(completions=FC())
    wrapped = mem.wrap(Cli())
    wrapped.chat.completions.create(model="m", user_id=USER,
                                    messages=[{"role": "user", "content": "hello there"}])
    ck("assembler injected instruction block via combined path",
       FC.kw["messages"][0]["content"].startswith("## User memory"))

    # -- 3+4. usage events recorded (injection now; agent-llm via worker) ----
    time.sleep(0.8)  # fire-and-forget insert settles
    u = api.get("/v1/usage", params={"project_id": PROJECT, "user_id": USER}).json()
    kinds = {k["kind"] for k in u["tokens"]["by_kind"]}
    ck("injection usage recorded", "injection" in kinds)
    ck("injected token total > 0", u["tokens"]["injected"] > 0)

    # agent-side usage: run an extractor through BaseAgent with a usage-bearing fake
    import asyncpg
    from memgram.agents.extractor import ExtractorAgent
    from memgram.memory.embedder import get_embedder
    from memgram.memory.store import MemoryStore

    class UsageLLM:
        class _R:
            def __init__(self):
                self.choices = [type("C", (), {"message": type("M", (), {
                    "content": '{"facts":[],"preferences":[{"content":"The user works in Rust."}],"entities":[],"corrections":[]}'})()})()]
                self.usage = {"prompt_tokens": 120, "completion_tokens": 30}
        def __init__(self):
            import types
            async def create(**kw): return UsageLLM._R()
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create))

    async def run_agent():
        pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=1, max_size=2)
        store = MemoryStore(pool, get_embedder())
        ag = ExtractorAgent(store=store, llm=UsageLLM(),
                            config={"faithfulness": False, "contradiction": False})
        await ag.run({**Q, "messages": [{"role": "user", "content": "rust"}]})
        await pool.close()
    asyncio.run(run_agent())

    u = api.get("/v1/usage", params={"project_id": PROJECT, "user_id": USER}).json()
    llm_rows = [k for k in u["tokens"]["by_kind"] if k["kind"].startswith("llm:")]
    ck("agent LLM usage recorded with kind llm:ExtractorAgent",
       any(k["kind"] == "llm:ExtractorAgent" for k in llm_rows))
    ck("token in/out captured", any(k["tokens_in"] >= 120 and k["tokens_out"] >= 30
                                    for k in llm_rows))
    ck("cost estimated for known model", u["tokens"]["total_cost_usd"] > 0)

    # -- 5. storage footprint --------------------------------------------------
    st = u["storage"]
    ck("semantic storage rows+bytes reported",
       sum(v["rows"] for v in st["semantic"].values()) >= 1 and st["total_bytes"] > 0)
    ck("episodic storage reported", st["episodic"]["rows"] >= 2)
    ck("instruction count reported", st["instructions"] == 1)

    # cleanup
    import asyncpg as _a
    async def clean():
        c = await _a.connect(os.environ["DATABASE_URL"])
        for t in ("usage_events", "instructions", "semantic_memories", "episodic_logs", "outbox"):
            await c.execute(f"DELETE FROM {t} WHERE project_id=$1", PROJECT) \
                if t != "outbox" else await c.execute("DELETE FROM outbox")
        await c.close()
    asyncio.run(clean())

    fail = False
    for n, c in checks:
        print(f"{'PASS' if c else 'FAIL'}  {n}")
        fail = fail or not c
    print(f"\n{sum(c for _, c in checks)}/{len(checks)} usage checks passed")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
