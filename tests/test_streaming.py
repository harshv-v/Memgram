"""Streaming support — pure logic, no network.

    python tests/test_streaming.py
"""
import asyncio
import sys
import time

from memgram.sdk.proxy import WrappedClient, _AsyncStream, _SyncStream, _delta_text

checks = []
def ck(n, c): checks.append((n, bool(c)))


def chunk(text):
    return {"choices": [{"delta": {"content": text}}]}


# ---- delta extraction -----------------------------------------------------------
ck("dict chunk delta", _delta_text(chunk("hi")) == "hi")
ck("empty delta safe", _delta_text({"choices": [{"delta": {}}]}) == "")
ck("garbage chunk safe", _delta_text(object()) == "")

# ---- sync stream: passthrough + accumulate + single fire --------------------------
done = []
s = _SyncStream(iter([chunk("Hel"), chunk("lo"), chunk("!")]), done.append)
got = [_delta_text(c) for c in s]
ck("chunks passed through in order", got == ["Hel", "lo", "!"])
ck("post-hook fired once with full text", done == ["Hello!"])
list(s.__iter__()) if False else None
s._finish()
ck("no double fire", done == ["Hello!"])

# early break still fires (developer stops reading mid-stream)
done2 = []
s2 = _SyncStream(iter([chunk("a"), chunk("b"), chunk("c")]), done2.append)
for c in s2:
    break
ck("early break still fires with partial text", done2 == ["a"])

# ---- async stream --------------------------------------------------------------
async def agen():
    for c in [chunk("Hey"), chunk(" you")]:
        yield c

async def run_async():
    done = []
    async def on_done(text): done.append(text)
    out = []
    async for c in _AsyncStream(agen(), on_done):
        out.append(_delta_text(c))
    return out, done

out, adone = asyncio.run(run_async())
ck("async chunks passed through", out == ["Hey", " you"])
ck("async post-hook fired with full text", adone == ["Hey you"])

# ---- end-to-end through the wrapped client ---------------------------------------
class FakeApi:
    called = None
    def ingest(self, user_id, agent_id, messages, text):
        FakeApi.called = (user_id, text)
class FakeAsm:
    def __init__(self, api): self._api = api
    def enrich(self, messages, user_id, agent_id):
        return [{"role": "system", "content": "INSTR"}] + list(messages)
class FC:
    def create(self, **kw):
        assert kw.get("stream")
        return iter([chunk("str"), chunk("eam")])
class FCli:
    def __init__(self):
        import types
        self.chat = types.SimpleNamespace(completions=FC())
class Cfg: agent_name = "a"

wc = WrappedClient(FCli(), FakeAsm(FakeApi()), Cfg())
stream = wc.chat.completions.create(model="m", stream=True, user_id="u1",
                                    messages=[{"role": "user", "content": "q"}])
text = "".join(_delta_text(c) for c in stream)
time.sleep(0.25)  # daemon thread
ck("wrapped stream yields chunks", text == "stream")
ck("wrapped stream post-hook ingested accumulated text",
   FakeApi.called == ("u1", "stream"))

fail = False
for n, c in checks:
    print(f"{'PASS' if c else 'FAIL'}  {n}")
    fail = fail or not c
print(f"\n{sum(c for _, c in checks)}/{len(checks)} streaming checks passed")
sys.exit(1 if fail else 0)
