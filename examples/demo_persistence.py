"""The 30-second demo — Week 1 definition of done.

Session 1: user states a preference; the app stores it as an instruction.
The process then EXITS and a brand-new process runs Session 2, where the user
asks an unrelated question with NO re-telling. The stored instruction is
auto-injected by the assembler, so the answer comes back concise and in Rust.

Run (with the API server already up):
    OPENAI_API_KEY=sk-...  MEMGRAM_API_KEY=mgram_dev_key  python examples/demo_persistence.py
"""
import os
import subprocess
import sys

import openai

from memgram import Memgram

USER_ID = "u1"
MODEL = os.environ.get("DEMO_MODEL", "gpt-4o-mini")


def make_client():
    mem = Memgram(
        api_key=os.environ.get("MEMGRAM_API_KEY", "mgram_dev_key"),
        agent_name="demo-agent",
        project_id="demo-project",
        api_base_url=os.environ.get("MEMGRAM_API_URL", "http://localhost:8000"),
    )
    client = mem.wrap(openai.OpenAI())  # the one wrapped line
    return mem, client


def session_1():
    print("=" * 70)
    print("SESSION 1 — user states preferences (pid %d)" % os.getpid())
    print("=" * 70)
    mem, client = make_client()

    msg = "I prefer concise answers and I work in Rust."
    print(f"\nuser> {msg}")
    r = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": msg}], user_id=USER_ID,
    )
    print(f"agent> {r.choices[0].message.content}")

    # Week 1: the app stores the preference via the instructions API.
    # (Week 2 replaces this manual step with the background extractor.)
    mem.add_instruction(USER_ID, "Always keep answers concise.", priority=1)
    mem.add_instruction(USER_ID, "The user works in Rust; give code examples in Rust.", priority=1)
    print("\n[stored 2 instructions via POST /v1/instructions]")
    print("[session 1 process exiting — all in-process state is lost]\n")


def session_2():
    print("=" * 70)
    print("SESSION 2 — brand new process (pid %d), NO re-telling" % os.getpid())
    print("=" * 70)
    _, client = make_client()

    msg = "Help me sort a list."
    print(f"\nuser> {msg}")
    r = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": msg}], user_id=USER_ID,
    )
    answer = r.choices[0].message.content
    print(f"agent> {answer}")

    rust = "rust" in answer.lower() or "fn " in answer or ".sort()" in answer
    concise = len(answer) < 1200
    print("\n--- verification ---")
    print(f"mentions Rust / Rust code : {rust}")
    print(f"concise (<1200 chars)     : {concise} ({len(answer)} chars)")
    print("DEMO " + ("PASSED [OK]" if rust and concise else "FAILED [X]"))
    sys.exit(0 if rust and concise else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "session2":
        session_2()
    else:
        session_1()
        # Real process restart — session 2 runs in a fresh interpreter.
        sys.exit(subprocess.call([sys.executable, __file__, "session2"]))
