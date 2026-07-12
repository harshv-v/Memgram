"""Multi-provider LLM layer — pure logic, no network.

    python tests/test_providers.py
"""
import asyncio
import json
import os
import sys
import time
import types

from memgram.llm import (AnthropicLLM, BRAINS, brain_spec, extract_json,
                         _with_json_instruction)

checks = []
def ck(n, c): checks.append((n, bool(c)))
def raises(f):
    try:
        f(); return False
    except Exception:
        return True

# ---- extract_json: fences, prose, nesting -----------------------------------
ck("clean json passes", extract_json('{"a": 1}') == '{"a": 1}')
ck("fenced json extracted", json.loads(extract_json('```json\n{"a": 1}\n```'))["a"] == 1)
ck("plain fence extracted", json.loads(extract_json('```\n{"a": 2}\n```'))["a"] == 2)
ck("prose-wrapped extracted",
   json.loads(extract_json('Sure! Here you go: {"a": 3} Hope that helps.'))["a"] == 3)
ck("nested braces balanced",
   json.loads(extract_json('x {"a": {"b": {"c": 1}}} y'))["a"]["b"]["c"] == 1)
ck("braces inside strings survive",
   json.loads(extract_json('{"s": "curly } brace {", "n": 4}'))["n"] == 4)
ck("no json -> original text back", extract_json("no braces here") == "no braces here")
ck("empty safe", extract_json("") == "")

# ---- brain presets ------------------------------------------------------------
os.environ.pop("MEMGRAM_BRAIN", None); os.environ.pop("MEMGRAM_LLM_BASE_URL", None)
ck("default brain = openai", brain_spec()["name"] == "openai")
os.environ["MEMGRAM_BRAIN"] = "deepseek"
ck("deepseek preset -> deepseek base_url + model",
   "deepseek" in brain_spec()["base_url"] and brain_spec()["fast"] == "deepseek-chat")
os.environ["MEMGRAM_BRAIN"] = "gemini"
ck("gemini preset -> google openai-compat endpoint",
   "generativelanguage.googleapis.com" in brain_spec()["base_url"])
os.environ["MEMGRAM_LLM_BASE_URL"] = "http://my-vllm:8000/v1"
ck("explicit base_url beats preset", brain_spec()["base_url"] == "http://my-vllm:8000/v1")
os.environ.pop("MEMGRAM_LLM_BASE_URL", None)
os.environ["MEMGRAM_BRAIN"] = "nope"
ck("unknown brain raises", raises(brain_spec))
os.environ["MEMGRAM_BRAIN"] = "anthropic"
ck("anthropic preset -> haiku fast tier", brain_spec()["fast"].startswith("claude-haiku"))
os.environ.pop("MEMGRAM_BRAIN", None)
ck("every preset has 5 fields", all(len(v) == 5 for v in BRAINS.values()))

# ---- model tier resolution follows the brain ------------------------------------
from memgram.agents.base import fast_model, quality_model
os.environ["MEMGRAM_BRAIN"] = "deepseek"
os.environ.pop("MEMGRAM_FAST_MODEL", None); os.environ.pop("MEMGRAM_QUALITY_MODEL", None)
ck("fast tier follows brain", fast_model() == "deepseek-chat")
os.environ["MEMGRAM_FAST_MODEL"] = "my-override"
ck("MEMGRAM_FAST_MODEL beats brain", fast_model() == "my-override")
os.environ.pop("MEMGRAM_FAST_MODEL", None); os.environ.pop("MEMGRAM_BRAIN", None)
ck("default quality = gpt-4o", quality_model() == "gpt-4o")

# ---- JSON instruction injection ---------------------------------------------------
msgs = [{"role": "system", "content": "You extract."}, {"role": "user", "content": "hi"}]
out = _with_json_instruction(msgs)
ck("json instruction appended to system", "valid JSON" in out[0]["content"])
ck("original messages not mutated", "valid JSON" not in msgs[0]["content"])
out2 = _with_json_instruction([{"role": "user", "content": "hi"}])
ck("no system -> one prepended", out2[0]["role"] == "system" and "valid JSON" in out2[0]["content"])

# ---- Anthropic adapter translation ------------------------------------------------
class FakeAnthropicMessages:
    last = None
    async def create(self, **kw):
        FakeAnthropicMessages.last = kw
        block = types.SimpleNamespace(text='```json\n{"facts": [{"content": "f1"}]}\n```')
        return types.SimpleNamespace(content=[block])

fake = types.SimpleNamespace(messages=FakeAnthropicMessages())
ad = AnthropicLLM(client=fake)
r = asyncio.run(ad.chat.completions.create(
    model="claude-haiku-4-5",
    messages=[{"role": "system", "content": "You extract long-term memories"},
              {"role": "user", "content": "I use Rust"}],
    response_format={"type": "json_object"}))
sent = FakeAnthropicMessages.last
ck("anthropic: system lifted to param", "extract long-term" in sent["system"])
ck("anthropic: json-strict appended", "valid JSON" in sent["system"])
ck("anthropic: only user/assistant turns in messages",
   all(m["role"] in ("user", "assistant") for m in sent["messages"]))
ck("anthropic: fenced reply -> clean json for BaseAgent",
   json.loads(r.choices[0].message.content)["facts"][0]["content"] == "f1")

# ---- BaseAgent runs end-to-end through the adapter --------------------------------
from memgram.agents.extractor import ExtractorAgent

class CaptureStore:
    def __init__(self): self.contents = []
    async def upsert_semantic(self, **kw):
        self.contents.append(kw["content"])
        return {"id": f"m{len(self.contents)}", "action": "created", "reinforcement_count": 1}

st = CaptureStore()
ag = ExtractorAgent(store=st, llm=ad, config={"faithfulness": False, "contradiction": False})
res = asyncio.run(ag.run({"project_id": "p", "agent_id": "a", "user_id": "u",
                          "messages": [{"role": "user", "content": "I use Rust"}]}))
ck("extractor pipeline works on anthropic adapter", st.contents == ["f1"])

# ---- parse_json fallback tolerates fences -----------------------------------------
from memgram.agents.base import BaseAgent
ck("parse_json tolerates fenced output",
   BaseAgent.parse_json('```json\n{"ok": true}\n```')["ok"] is True)

# ---- wrap() detection ---------------------------------------------------------------
from memgram import Memgram
from memgram.sdk.anthropic_proxy import AnthropicWrappedClient

mem = Memgram(api_key="k", agent_name="a", api_base_url="http://localhost:9")  # no server needed
class FakeOpenAIClient:
    chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: None))
class FakeAnthropicClient:
    def __init__(self):
        self.messages = types.SimpleNamespace(create=lambda **kw: None)
ck("openai-shaped client -> OpenAI proxy", hasattr(mem.wrap(FakeOpenAIClient()), "chat"))
ck("anthropic-shaped client -> Anthropic proxy",
   isinstance(mem.wrap(FakeAnthropicClient()), AnthropicWrappedClient))
ck("unknown client shape raises", raises(lambda: mem.wrap(object())))

# ---- anthropic wrap: system merge + id strip + post-hook --------------------------
class RecApi:
    ingested = None
    def get_instructions(self, u, a):
        return [{"priority": 1, "content": "Always answer in Rust."}]
    def search_memories(self, u, a, q, limit=5):
        return [{"content": "The user works at Acme."}]
    def ingest(self, u, a, msgs, text): RecApi.ingested = (u, a, text)

class RecMessages:
    last = None
    def create(self, **kw):
        RecMessages.last = kw
        block = types.SimpleNamespace(text="fn main() {}")
        return types.SimpleNamespace(content=[block])

class Cfg2:
    agent_name = "a"
    features = {"semantic": True}

wrapped = AnthropicWrappedClient(
    types.SimpleNamespace(messages=RecMessages()), RecApi(), Cfg2())
resp = wrapped.messages.create(
    model="claude-haiku-4-5", system="You are a coding assistant.", max_tokens=100,
    messages=[{"role": "user", "content": "sort a list"}], user_id="u7")
time.sleep(0.2)  # daemon-thread post-hook
sent = RecMessages.last
ck("anthropic wrap: instructions first in system",
   sent["system"].startswith("## User memory"))
ck("anthropic wrap: dev system preserved after instructions",
   "coding assistant" in sent["system"]
   and sent["system"].index("User memory") < sent["system"].index("coding assistant"))
ck("anthropic wrap: memories after dev system",
   "Acme" in sent["system"] and sent["system"].index("coding assistant") < sent["system"].index("Acme"))
ck("anthropic wrap: ids stripped", "user_id" not in sent and "agent_id" not in sent)
ck("anthropic wrap: other kwargs pass through", sent["max_tokens"] == 100)
ck("anthropic wrap: post-hook ingested with text",
   RecApi.ingested == ("u7", "a", "fn main() {}"))

fail = False
for n, c in checks:
    print(f"{'PASS' if c else 'FAIL'}  {n}")
    fail = fail or not c
print(f"\n{sum(c for _, c in checks)}/{len(checks)} provider checks passed")
sys.exit(1 if fail else 0)
