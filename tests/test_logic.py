"""Pure-logic suite — no Postgres, no Redis, no network. Covers the wiring that
the integration tests can't isolate: prompt assembly + injection order, the
post-hook, and every agent's JSON parsing / rendering.

    MEMGRAM_FAKE_LLM=1 python tests/test_logic.py
"""
import asyncio
import json
import sys
import time

checks = []
def ck(n, c): checks.append((n, bool(c)))
def raises(f):
    try:
        f(); return False
    except Exception:
        return True


# ---- assembler: formatting + strict injection order --------------------------
from memgram.sdk import assembler as A

ck("instructions formatted w/ priority",
   "[priority 1]" in (A._format_instructions([{"priority": 1, "content": "x"}]) or ""))
ck("empty instructions -> None", A._format_instructions([]) is None)
ck("semantic formatted", "- uses Rust" in (A._format_semantic([{"content": "uses Rust"}]) or ""))

msgs = [{"role": "system", "content": "dev"}, {"role": "user", "content": "hi"}]
out = A._inject(msgs, "INSTR", "SEM")
# Spec order: instructions -> dev system prompt -> semantic -> rest
ck("inject: instructions first", out[0]["content"] == "INSTR")
ck("inject: dev system prompt kept in place", out[1]["content"] == "dev")
ck("inject: semantic AFTER dev system prompt", out[2]["content"] == "SEM")
ck("inject: conversation preserved last", out[3]["content"] == "hi")
out2 = A._inject([{"role": "user", "content": "hi"}], "INSTR", "SEM")
ck("inject: no dev system -> instr, sem, user", [m["content"] for m in out2] == ["INSTR", "SEM", "hi"])
ck("inject: nothing -> unchanged object", A._inject(msgs, None, None) is msgs)
ck("last-user picks latest",
   A._last_user_text([{"role": "user", "content": "a"}, {"role": "user", "content": "c"}]) == "c")


# ---- client._clean: never re-log our own injected blocks ---------------------
from memgram.sdk.client import _clean

cleaned = _clean([
    {"role": "system", "content": "## User memory — standing instructions\nx"},
    {"role": "system", "content": "## Relevant memory — y"},
    {"role": "system", "content": "real dev prompt"},
    {"role": "user", "content": "hello"}])
ck("_clean strips memory blocks, keeps dev+user",
   [m["content"] for m in cleaned] == ["real dev prompt", "hello"])


# ---- proxy: enrich + id strip + same object + fire-and-forget post-hook ------
from memgram.sdk.proxy import WrappedClient, _response_text

class FakeApi:
    called = None
    def ingest(self, user_id, agent_id, messages, text):
        FakeApi.called = (user_id, agent_id, text)
class FakeAsm:
    def __init__(self, api): self._api = api
    def enrich(self, messages, user_id, agent_id):
        return [{"role": "system", "content": "INSTR"}] + list(messages)
class FC:
    kw = None
    def create(self, **kw):
        FC.kw = kw
        return {"id": "R", "choices": [{"message": {"content": "hi there"}}]}
class FChat:
    def __init__(self): self.completions = FC()
class FCli:
    def __init__(self): self.chat = FChat()
class Cfg: agent_name = "a"

api = FakeApi()
wc = WrappedClient(FCli(), FakeAsm(api), Cfg())
resp = wc.chat.completions.create(model="m", user_id="u9", agent_id="ag",
                                  messages=[{"role": "user", "content": "q"}])
time.sleep(0.25)  # let the daemon-thread post-hook run
ck("enrich injected + stripped ids",
   FC.kw["messages"][0]["content"] == "INSTR" and "user_id" not in FC.kw and "agent_id" not in FC.kw)
ck("response object unchanged", resp["id"] == "R")
ck("_response_text extracts assistant text", _response_text(resp) == "hi there")
ck("post-hook ingest fired w/ user+agent",
   FakeApi.called and FakeApi.called[0] == "u9" and FakeApi.called[1] == "ag")


# ---- agent parsers (pure JSON) ----------------------------------------------
from memgram.agents.extractor import ExtractorAgent
p = ExtractorAgent(store=None, llm=None).parse_output(json.dumps(
    {"facts": [{"content": "f"}], "preferences": [], "entities": [], "corrections": [{"content": "c"}]}))
ck("extractor types facts/corrections",
   p["facts"][0]["memory_type"] == "fact" and p["corrections"][0]["memory_type"] == "correction")

from memgram.agents.reflection import ReflectionAgent
r = ReflectionAgent(store=None, llm=None).parse_output(json.dumps(
    {"insights": [{"content": "i", "memory_type": "preference"}, {"content": "x", "memory_type": "bad"}]}))
ck("reflection drops invalid insight type", len(r["insights"]) == 1)

from memgram.agents.proposer import ProposerAgent
pp = ProposerAgent(store=None, llm=None)
ck("proposer parses instruction", pp.parse_output(json.dumps({"instruction": "Do X", "priority": 1}))["instruction"] == "Do X")
ck("proposer rejects empty instruction", raises(lambda: pp.parse_output('{"instruction":""}')))

from memgram.agents.summarizer import SummarizerAgent
sm = SummarizerAgent(store=None, llm=None)
so = sm.parse_output(json.dumps({"key_decisions": ["d1"], "facts_established": [],
                                 "errors_corrected": [], "preferences_revealed": [], "open_threads": []}))
ck("summarizer parses sections", so["key_decisions"] == ["d1"])
ck("summarizer render has header+item", "Session summary" in sm.render(so) and "d1" in sm.render(so))
ck("summarizer render empty -> ''",
   sm.render({k: [] for k in ['key_decisions', 'facts_established', 'errors_corrected',
                              'preferences_revealed', 'open_threads']}) == "")


# ---- FakeLLM routing (the test brain) ---------------------------------------
from memgram.testing import FakeLLM
llm = FakeLLM()
async def call(system, user):
    rr = await llm.chat.completions.create(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return json.loads(rr.choices[0].message.content)

extr = asyncio.run(call("You extract long-term memories", "I love Rust and concise answers"))
ck("FakeLLM extractor -> 2 prefs (rust+concise)", len(extr["preferences"]) == 2)
summ = asyncio.run(call("compress a long conversation", "rust"))
ck("FakeLLM summarizer -> has sections", "preferences_revealed" in summ)


# ---- presets + layered config (section 9) -----------------------------------
from memgram.presets import resolve
d = resolve(None)
ck("defaults: semantic on, habit 7", d["features"]["semantic"] and d["decay"]["habit_threshold"] == 7)
m = resolve("minimal")
ck("preset minimal: reflection+decay off", not m["features"]["reflection"] and not m["features"]["decay"])
ck("preset minimal: instructions still on", m["features"]["instructions"])
c = resolve("coding")
ck("preset coding: procedural on, habit 10", c["features"]["procedural"] and c["decay"]["habit_threshold"] == 10)
o = resolve("chatbot", {"memory_budget": 1234, "decay": {"habit_threshold": 3}})
ck("overrides beat preset (budget + nested habit)", o["memory_budget"] == 1234 and o["decay"]["habit_threshold"] == 3)
ck("unknown preset raises", raises(lambda: resolve("nope")))
ck("bad sharing_scope raises", raises(lambda: resolve(None, {"sharing_scope": "weird"})))

from memgram.sdk.config import MemgramConfig
mc = MemgramConfig(api_key="k", agent_name="a", preset="minimal")
ck("config exposes resolved settings", mc.features["semantic"] is False and mc.memory_budget == 4000)

# ---- model tiers ------------------------------------------------------------
from memgram.agents.base import fast_model, quality_model
ck("extractor uses fast model", ExtractorAgent(store=None, llm=None).model == fast_model())
ck("reflection uses quality model", ReflectionAgent(store=None, llm=None).model == quality_model())
ck("explicit model config wins",
   ExtractorAgent(store=None, config={"model": "x"}, llm=None).model == "x")

# ---- summarizer auto-trigger estimator --------------------------------------
from memgram.api.routes.ingest import _est_tokens
ck("token estimate ~ chars/4", _est_tokens([{"role": "user", "content": "x" * 400}]) == 100)


# ---- operation-selection integration (contradiction v2) ----------------------
# One bounded op per new fact: ADD / UPDATE(target) / DELETE(target) / NOOP.
class OpStore:
    def __init__(self, cands):
        self.cands, self._n = cands, 0
        self.upserts, self.updates, self.supersedes, self.searches = [], [], [], 0
    async def find_similar_active(self, project_id, agent_id, user_id, content, limit=8):
        self.searches += 1
        return list(self.cands)
    async def upsert_semantic(self, **kw):
        self._n += 1
        self.upserts.append(kw["content"])
        return {"id": f"new{self._n}", "action": "created", "reinforcement_count": 1}
    async def update_semantic(self, memory_id, content):
        self.updates.append((memory_id, content))
        return {"id": memory_id, "action": "updated", "reinforcement_count": 2}
    async def supersede(self, old_id, new_id):
        self.supersedes.append((old_id, new_id))

def run_integration(facts, cands, contradiction=True):
    st = OpStore(cands)
    ag = ExtractorAgent(store=st, llm=FakeLLM(),
                        config={"contradiction": contradiction, "faithfulness": False})
    job = {"project_id": "p", "agent_id": "a", "user_id": "u",
           "messages": [{"role": "user", "content": "ctx"}]}
    result = {"facts": [{"content": f, "memory_type": "fact"} for f in facts]}
    asyncio.run(ag.on_success(job, result))
    return st

st = run_integration(["The user lives in Munich."],
                     [{"id": "old1", "content": "The user lives in Berlin.", "dist": 0.3}])
ck("op DELETE: new fact stored", st.upserts == ["The user lives in Munich."])
ck("op DELETE: old memory soft-invalidated by new id", st.supersedes == [("old1", "new1")])

st = run_integration(["The user is a tech lead."],
                     [{"id": "m1", "content": "The user works at Acme.", "dist": 0.2}])
ck("op UPDATE: merged in place, id kept",
   st.updates == [("m1", "The user works at Acme as a tech lead.")])
ck("op UPDATE: no new row, nothing archived", st.upserts == [] and st.supersedes == [])

st = run_integration(["The user lives in Berlin."],
                     [{"id": "m1", "content": "The user lives in Berlin.", "dist": 0.16}])
ck("op NOOP: restated fact dropped",
   st.upserts == [] and st.updates == [] and st.supersedes == [])

st = run_integration(["The user has a dog named Rex."],
                     [{"id": "m1", "content": "The user works at Acme.", "dist": 0.5}])
ck("op ADD: unrelated candidate untouched", st.upserts and st.supersedes == [] and st.updates == [])

st = run_integration(["The user said bogus target."],
                     [{"id": "m1", "content": "The user works at Acme.", "dist": 0.5}])
ck("op safety: target outside candidate set -> ADD, no archive",
   st.upserts and st.supersedes == [])

st = run_integration(["The user has a cat."], [])
ck("op: no candidates -> direct ADD (no LLM decision)", st.upserts == ["The user has a cat."])

# batch safety: this turn's facts can't supersede each other
st = run_integration(["The user has a parrot.", "The user lives in Munich."],
                     [{"id": "new1", "content": "The user lives in Berlin.", "dist": 0.3}])
ck("op batch safety: same-turn id excluded from targets",
   len(st.upserts) == 2 and st.supersedes == [])

st = run_integration(["The user lives in Munich."],
                     [{"id": "old1", "content": "The user lives in Berlin.", "dist": 0.3}],
                     contradiction=False)
ck("contradiction off: plain store, no search",
   st.upserts == ["The user lives in Munich."] and st.searches == 0 and st.supersedes == [])

# per-job override beats the worker default (the eval's single-stack A/B switch)
def run_with_job_override(worker_default, job_override):
    st = OpStore([{"id": "old1", "content": "The user lives in Berlin.", "dist": 0.3}])
    ag = ExtractorAgent(store=st, llm=FakeLLM(),
                        config={"contradiction": worker_default, "faithfulness": False})
    job = {"project_id": "p", "agent_id": "a", "user_id": "u",
           "messages": [{"role": "user", "content": "ctx"}],
           "contradiction": job_override}
    asyncio.run(ag.on_success(job, {"facts": [
        {"content": "The user lives in Munich.", "memory_type": "fact"}]}))
    return st

st = run_with_job_override(worker_default=False, job_override=True)
ck("job override ON beats worker default off (v2 ran)",
   st.searches == 1 and st.supersedes == [("old1", "new1")])
st = run_with_job_override(worker_default=True, job_override=False)
ck("job override OFF beats worker default on (plain store)",
   st.searches == 0 and st.supersedes == [])


fail = False
for n, c in checks:
    print(f"{'PASS' if c else 'FAIL'}  {n}")
    fail = fail or not c
print(f"\n{sum(c for _, c in checks)}/{len(checks)} logic checks passed")
sys.exit(1 if fail else 0)
