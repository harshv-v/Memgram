"""Real agents on Memgram — the dogfood.

Two agents you can actually talk to:
  ChatAgent   - a conversational assistant; every exchange feeds memory.
  CoderAgent  - a tool-using agent (runs Python); tool failures become
                procedural memory (use the `coding` preset to enable).

Both work with a real key OR with the zero-cost fake stack — the memory
mechanics are identical either way.
"""
import contextlib
import io
import json
import os

import openai

from memgram import Memgram


def make_mem(agent_name: str, preset: str | None = None) -> Memgram:
    return Memgram(
        api_key=os.environ.get("MEMGRAM_API_KEY", "mgram_dev_key"),
        agent_name=agent_name,
        project_id=os.environ.get("MEMGRAM_PROJECT", "agents-demo"),
        api_base_url=os.environ.get("MEMGRAM_API_URL", "http://localhost:8000"),
        preset=preset,
    )


class ChatAgent:
    """Minimal conversational agent. Session history lives in RAM (and dies
    with the process); everything durable lives in Memgram."""

    def __init__(self, agent_name: str, user_id: str, system: str | None = None,
                 model: str | None = None, preset: str | None = None):
        self.mem = make_mem(agent_name, preset)
        self.client = self.mem.wrap(openai.OpenAI())
        self.user_id = user_id
        self.model = model or os.environ.get("DEMO_MODEL", "gpt-4o-mini")
        self.history: list[dict] = []
        if system:
            self.history.append({"role": "system", "content": system})

    def chat(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        r = self.client.chat.completions.create(
            model=self.model, messages=list(self.history), user_id=self.user_id)
        answer = r.choices[0].message.content
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def new_session(self):
        """Simulate closing the app: RAM history gone, memory persists."""
        self.history = [m for m in self.history if m["role"] == "system"][:1]


def _run_python(code: str) -> str:
    """Toy sandbox for the demo agent. NOT a security boundary — demo only."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": {"print": print, "range": range, "len": len,
                                         "sum": sum, "min": min, "max": max}})
        return buf.getvalue() or "(no output)"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


_TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Execute a short Python snippet and return its stdout.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}}, "required": ["code"]},
    },
}]


class CoderAgent(ChatAgent):
    """Tool-using agent: the model can run Python. The full tool loop rides
    through the wrapped client, so tool calls + results land in episodic
    memory and (with the coding preset) distill into procedural lessons."""

    def __init__(self, user_id: str):
        super().__init__("coder", user_id, preset="coding",
                         system="You are a coding assistant. Use run_python when execution helps.")

    def chat(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        for _ in range(5):  # bounded tool loop
            r = self.client.chat.completions.create(
                model=self.model, messages=list(self.history),
                tools=_TOOLS, user_id=self.user_id)
            msg = r.choices[0].message
            if not getattr(msg, "tool_calls", None):
                self.history.append({"role": "assistant", "content": msg.content})
                return msg.content
            self.history.append(msg)
            for tc in msg.tool_calls:
                out = _run_python(json.loads(tc.function.arguments).get("code", ""))
                self.history.append({"role": "tool", "tool_call_id": tc.id, "content": out})
        return "(tool loop limit reached)"


if __name__ == "__main__":
    import sys
    user = os.environ.get("DEMO_USER", "harsha")
    agent = CoderAgent(user) if "--coder" in sys.argv else ChatAgent("assistant", user)
    print(f"[{type(agent).__name__}] user={user} — type 'quit' to exit, 'new' for a fresh session")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line == "quit":
            break
        if line == "new":
            agent.new_session()
            print("(new session — RAM history cleared; memory persists)")
            continue
        if line:
            print(f"agent> {agent.chat(line)}")
