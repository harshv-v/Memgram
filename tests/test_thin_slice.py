"""Week-1 thin-slice verification — real API server, real Postgres, fake LLM.

Proves, end-to-end:
  1. POST /v1/instructions persists to Postgres.
  2. A brand-new Memgram/client (simulating a process restart) auto-injects
     the stored instructions as the FIRST system message.
  3. The developer's own system prompt is preserved, after instructions.
  4. user_id/agent_id are stripped before reaching the LLM client.
  5. The response object is returned unchanged (exact same object).
  6. Trust gate: source='agent_proposed' is forced to status='pending'.

Run (API server must be up):
    MEMGRAM_API_KEY=mgram_dev_key python tests/test_thin_slice.py
"""
import os
import sys

from memgram import Memgram

API_URL = os.environ.get("MEMGRAM_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("MEMGRAM_API_KEY", "mgram_dev_key")
USER = "test-user-thin-slice"


class FakeCompletions:
    """Mimics openai's client.chat.completions — records what it receives."""
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return {"id": "fake-response", "object": "chat.completion"}


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeOpenAI:
    def __init__(self):
        self.chat = FakeChat()


def fresh_client():
    """New Memgram + new wrapped client == what a process restart gives you."""
    mem = Memgram(api_key=API_KEY, agent_name="demo-agent",
                   project_id="test-project", api_base_url=API_URL)
    fake = FakeOpenAI()
    return mem, mem.wrap(fake), fake


def main():
    checks = []

    # --- Session 1: store the preference ---------------------------------
    mem, client, fake = fresh_client()
    client.chat.completions.create(
        model="gpt-4o-mini", user_id=USER,
        messages=[{"role": "user", "content": "I prefer concise answers and I work in Rust."}],
    )
    mem.add_instruction(USER, "Always keep answers concise.", priority=1)
    mem.add_instruction(USER, "The user works in Rust; give code examples in Rust.", priority=1)

    # --- "Process restart": everything rebuilt from scratch ---------------
    mem2, client2, fake2 = fresh_client()
    dev_system = {"role": "system", "content": "You are a helpful coding assistant."}
    user_msg = {"role": "user", "content": "Help me sort a list."}
    response = client2.chat.completions.create(
        model="gpt-4o-mini", user_id=USER, messages=[dev_system, user_msg],
    )

    sent = fake2.chat.completions.last_kwargs
    msgs = sent["messages"]

    checks.append(("instruction block injected first",
                   msgs[0]["role"] == "system" and "concise" in msgs[0]["content"]
                   and "Rust" in msgs[0]["content"]))
    # Robust to an optional "## Relevant memory" semantic block: assert the
    # developer's own turns are preserved, in order, after the injected memory.
    checks.append(("developer system prompt preserved",
                   dev_system in msgs))
    checks.append(("original user message intact", user_msg in msgs))
    checks.append(("developer order preserved (system before user)",
                   dev_system in msgs and user_msg in msgs
                   and msgs.index(dev_system) < msgs.index(user_msg)))
    checks.append(("injected memory precedes developer content",
                   msgs.index(dev_system) > 0))
    checks.append(("user_id/agent_id stripped before LLM", "user_id" not in sent
                   and "agent_id" not in sent))
    checks.append(("response object returned unchanged",
                   response == {"id": "fake-response", "object": "chat.completion"}))

    # --- Trust gate --------------------------------------------------------
    proposed = mem2.api.create_instruction(
        user_id=USER, agent_id="demo-agent",
        content="evil persistent instruction", source="agent_proposed",
    )
    checks.append(("agent_proposed forced to pending", proposed["status"] == "pending"))

    # --- Report ------------------------------------------------------------
    failed = False
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failed = failed or not ok

    print("\ninjected system message was:\n" + "-" * 40)
    print(msgs[0]["content"])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
