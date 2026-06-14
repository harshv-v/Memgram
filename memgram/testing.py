"""Fakes for tests and zero-cost local dev. MEMGRAM_FAKE_LLM=1 switches the
worker to FakeLLM; the full pipeline (extract → reflect → propose) runs with
no external calls. The fake answers are keyed off the system prompt."""
import json


class _Msg:
    def __init__(self, content): self.content = content


class _Choice:
    def __init__(self, content): self.message = _Msg(content)


class _Resp:
    def __init__(self, content): self.choices = [_Choice(content)]


class _FakeCompletions:
    async def create(self, model=None, messages=None, response_format=None, **kw):
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""
        if "fact-checker" in system:
            # faithfulness pass: in tests, treat every candidate as supported
            return _Resp(json.dumps({"supported": list(range(50))}))
        if "extract long-term memories" in system:
            out = {"facts": [], "preferences": [], "entities": [], "corrections": []}
            low = user.lower()
            if "rust" in low:
                out["preferences"].append({"content": "The user works in Rust."})
            if "concise" in low:
                out["preferences"].append({"content": "The user prefers concise answers."})
            if "acme" in low:
                out["facts"].append({"content": "The user works at Acme."})
            if "wrong" in low or "no," in low or "not what i" in low:
                out["corrections"].append({"content": "The assistant gave a wrong answer the user corrected."})
            return _Resp(json.dumps(out))
        if "compress a long conversation" in system:
            out = {"key_decisions": [], "facts_established": [],
                   "errors_corrected": [], "preferences_revealed": [],
                   "open_threads": []}
            if "rust" in user.lower():
                out["preferences_revealed"].append("The user works in Rust.")
            return _Resp(json.dumps(out))
        if "reflection process" in system:
            insights = []
            if "rust" in user.lower():
                insights.append({"content": "The user consistently works in Rust.",
                                 "memory_type": "preference"})
            return _Resp(json.dumps({"insights": insights}))
        if "behavioural pattern" in system:
            return _Resp(json.dumps({
                "instruction": "Always give code examples in Rust.", "priority": 1}))
        return _Resp(json.dumps({}))


class _FakeChat:
    def __init__(self): self.completions = _FakeCompletions()


class FakeLLM:
    def __init__(self): self.chat = _FakeChat()
