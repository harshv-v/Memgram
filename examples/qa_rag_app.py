"""A tiny Q&A assistant built on Memgram — shows the memory layer doing
retrieval-augmented generation for you.

What it demonstrates, end to end against the live stack:
  1. A user has a multi-turn conversation; facts/preferences they mention are
     extracted into semantic memory by the worker (no code asks for that).
  2. A *brand-new* session (fresh client, no history passed) recalls those
     memories automatically and answers in light of them.
  3. A *different* user asking the same question gets none of the first user's
     memory — tenant isolation.
  4. Reinforcement: re-asking related questions bumps a memory's access count.

Run (stack already up via `docker compose up`):
    OPENAI_API_KEY=sk-...  python examples/qa_rag_app.py

The script then prints exactly what Memgram stored, via the public API.
"""
import os
import sys
import time

import openai

from memgram import Memgram

# LLM replies can contain box-drawing / unicode glyphs; the Windows console
# defaults to cp1252 and would crash on them. Make stdout tolerant.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

API_BASE = os.environ.get("MEMGRAM_API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("MEMGRAM_API_KEY", "mgram_dev_key")
EXTRACT_WAIT = float(os.environ.get("QA_EXTRACT_WAIT", "6"))  # let the worker run


def make_client(agent="qa-assistant"):
    mem = Memgram(api_key=API_KEY, agent_name=agent,
                  project_id="qa-demo", api_base_url=API_BASE)
    return mem, mem.wrap(openai.OpenAI())


def ask(client, user_id, question):
    print(f"\n[{user_id}] Q: {question}")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        user_id=user_id,
    )
    answer = resp.choices[0].message.content.strip()
    preview = answer if len(answer) <= 320 else answer[:320] + " …[truncated]"
    print(f"[{user_id}] A: {preview}")
    return answer


def show_memory(mem, user_id):
    """Read back what Memgram stored for this user, via the public API."""
    api = mem.api._sync
    h = {"Authorization": f"Bearer {API_KEY}"}
    params = {"project_id": "qa-demo", "agent_id": "qa-assistant", "user_id": user_id}

    sem = api.get("/v1/memory", params={**params, "limit": 50}, headers=h).json()
    print(f"\n  semantic memories for {user_id} ({len(sem['memories'])}):")
    for m in sem["memories"]:
        print(f"    - [{m['memory_type']}/{m['memory_tier']} x{m['reinforcement_count']}] "
              f"{m['content']}")
    return sem["memories"]


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY to run the live Q&A demo.")

    print("=" * 70)
    print("SESSION 1 — Alice has a few exchanges with the assistant")
    print("=" * 70)
    mem, client = make_client()
    ask(client, "alice", "I'm building a REST API in Python with FastAPI. "
                          "Keep your answers short. How do I add a health endpoint?")
    ask(client, "alice", "How do I read an environment variable?")

    print(f"\n... letting the worker extract (waiting {EXTRACT_WAIT}s) ...")
    time.sleep(EXTRACT_WAIT)
    show_memory(mem, "alice")

    print("\n" + "=" * 70)
    print("SESSION 2 — BRAND NEW client/process state. No history passed.")
    print("Memgram should recall Alice works in Python/FastAPI and wants brevity.")
    print("=" * 70)
    mem2, client2 = make_client()
    a = ask(client2, "alice", "What's a good way to validate request bodies?")
    py = any(k in a.lower() for k in ("pydantic", "fastapi", "python"))
    print(f"\n  -> recalled Alice's stack (mentions pydantic/fastapi/python): {py}")

    print("\n" + "=" * 70)
    print("ISOLATION — Bob asks the same thing; must NOT inherit Alice's memory")
    print("=" * 70)
    ask(client2, "bob", "What's a good way to validate request bodies?")
    time.sleep(EXTRACT_WAIT)
    bob_mem = show_memory(mem2, "bob")
    alice_leak = any("fastapi" in m["content"].lower() or "python" in m["content"].lower()
                     for m in bob_mem)
    print(f"\n  -> Bob's memory contains Alice's stack: {alice_leak}  "
          f"(should be False)")

    print("\n" + "=" * 70)
    print("REINFORCEMENT — re-asking related questions bumps access counts")
    print("=" * 70)
    ask(client2, "alice", "Remind me how to structure a FastAPI project?")
    time.sleep(EXTRACT_WAIT)
    final = show_memory(mem2, "alice")

    print("\n" + "=" * 70)
    reinforced = any(m["reinforcement_count"] > 1 for m in final)
    ok = py and not alice_leak
    print(f"recall works: {py} | isolation holds: {not alice_leak} | "
          f"reinforcement seen: {reinforced}")
    print("Q&A RAG DEMO " + ("PASSED [OK]" if ok else "CHECK OUTPUT [!]"))


if __name__ == "__main__":
    main()
