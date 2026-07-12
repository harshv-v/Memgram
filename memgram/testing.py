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
        if "PROCEDURAL lessons" in system:
            procs = []
            if "tool" in user.lower() or "(" in user:
                procs.append({"content": "The search tool fails without a date filter; always include one."})
            return _Resp(json.dumps({"procedures": procs}))
        if "integrate ONE new fact" in system:
            # operation selection: deterministic ops for the logic tests.
            # Parse "NEW FACT:\n<content>" and "- id=<id>: <text>" candidate lines.
            new_fact = user.split("<new_fact>")[-1].split("</new_fact>")[0].strip().lower()
            cands = []
            for line in user.splitlines():
                line = line.strip()
                if line.startswith("- id="):
                    cid, txt = line[5:].split(":", 1)
                    cands.append((cid.strip(), txt.strip().lower()))
            for cid, txt in cands:
                if txt == new_fact:  # restating an existing memory -> NOOP
                    return _Resp(json.dumps({"op": "NOOP", "target_id": None, "content": None}))
            if "munich" in new_fact:  # city change -> DELETE the old-city memory
                for cid, txt in cands:
                    if "berlin" in txt:
                        return _Resp(json.dumps({"op": "DELETE", "target_id": cid, "content": None}))
            if "tech lead" in new_fact:  # same attribute refined -> UPDATE (merge)
                for cid, txt in cands:
                    if "acme" in txt:
                        return _Resp(json.dumps({
                            "op": "UPDATE", "target_id": cid,
                            "content": "The user works at Acme as a tech lead."}))
            if "bogus target" in new_fact:  # invalid target: extractor must fall back to ADD
                return _Resp(json.dumps({"op": "DELETE", "target_id": "no-such-id", "content": None}))
            return _Resp(json.dumps({"op": "ADD", "target_id": None, "content": None}))
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
            if "berlin" in low:
                out["facts"].append({"content": "The user lives in Berlin."})
            if "munich" in low:
                out["facts"].append({"content": "The user lives in Munich."})
            if "tech lead" in low:
                out["facts"].append({"content": "The user is a tech lead."})
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
