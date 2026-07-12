"""Simulated life — three weeks of real usage in ~3 minutes (real LLM).

Drives the example agents through a scripted arc that exercises every memory
behaviour, printing what the memory layer knows between phases so you can
watch memory FORM, CORRECT, SUPERSEDE, and HARDEN in real time. Run with the
dashboard open (localhost:3000) for the full effect.

    OPENAI_API_KEY=sk-... python examples/agents/simulated_life.py
"""
import os
import time

import httpx

from memory_agents import ChatAgent, make_mem

USER = "sim-harsha"
BASE = os.environ.get("MEMGRAM_API_URL", "http://localhost:8000")
KEY = os.environ.get("MEMGRAM_API_KEY", "mgram_dev_key")
api = httpx.Client(base_url=BASE, headers={"Authorization": f"Bearer {KEY}"}, timeout=30)


def show_memory(title: str):
    time.sleep(4)  # let the worker settle
    r = api.get("/v1/memory", params={
        "project_id": os.environ.get("MEMGRAM_PROJECT", "agents-demo"),
        "agent_id": "assistant", "user_id": USER}).json()
    print(f"\n──── memory after: {title} " + "─" * max(1, 40 - len(title)))
    for m in r["memories"][:10]:
        print(f"  [{m['memory_tier']:<8}] x{m['reinforcement_count']}  {m['content'][:70]}")
    print()


def main():
    a = ChatAgent("assistant", USER)

    print("== WEEK 1: the user introduces themselves ==")
    print("agent>", a.chat("Hey! I'm a backend dev, I work mostly in Rust, and I "
                           "like short answers.")[:120])
    print("agent>", a.chat("I live in Berlin and work at Acme on the payments team.")[:120])
    show_memory("introductions")

    print("== the user corrects a mistake (corrections persist longer) ==")
    print("agent>", a.chat("No, that's wrong - I said I prefer SHORT answers. "
                           "Please keep it brief.")[:120])
    show_memory("a correction")

    print("== WEEK 2: fresh session — no re-telling ==")
    a.new_session()
    print("agent>", a.chat("What language should I use for a quick CLI tool?")[:200])
    print("   (should assume Rust, briefly — recalled, not re-told)")

    print("\n== life changes: the user moves (contradiction: latest must win) ==")
    print("agent>", a.chat("Big news - I moved from Berlin to Munich last week!")[:120])
    show_memory("the move (Berlin should be superseded if MEMGRAM_CONTRADICTION=1)")

    print("== WEEK 3: repetition becomes habit ==")
    for i in range(6):
        a.new_session()
        a.chat(f"Quick Rust question #{i}: how do I read a file?")
    time.sleep(6)
    props = api.get("/v1/proposals", params={
        "project_id": os.environ.get("MEMGRAM_PROJECT", "agents-demo"),
        "agent_id": "assistant", "user_id": USER}).json()["proposals"]
    print(f"\npending habit proposals: {len(props)}")
    for p in props:
        print(f"  PROPOSED: {p['content']!r}  -> approve in the dashboard (or POST /approve)")

    print("\n== your memory is a file: export it ==")
    exp = api.get(f"/v1/memory/export/{USER}", params={
        "project_id": os.environ.get("MEMGRAM_PROJECT", "agents-demo")}).json()
    n = sum(len(v) for v in exp.values())
    print(f"exported {n} rows across {list(exp)} — portable to any other Memgram.")

    mem = make_mem("assistant")
    print("\nDone. Open the dashboard to browse tiers/provenance, approve the "
          "proposal, then ask the agent anything — the habit is now permanent.")
    _ = mem  # (kept for symmetry; the dashboard does the rest)


if __name__ == "__main__":
    main()
