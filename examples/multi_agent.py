"""Multi-agent memory: "this is agent A, this is agent B" — handled internally.

Two agents share one user. Memory is ISOLATED BY DEFAULT (each agent's memories
are private). Sharing is OPT-IN: promote a memory to `global` and every agent of
that user can see it. This shows a support agent learning a safety-critical fact
and sharing it, while keeping chit-chat private.

Run (stack up):
    OPENAI_API_KEY=sk-...  python examples/multi_agent.py
"""
import os
import sys
import time
import uuid

import httpx

API = os.environ.get("MEMGRAM_API_BASE_URL", "http://localhost:8000")
KEY = os.environ.get("MEMGRAM_API_KEY", "mgram_dev_key")
H = {"Authorization": f"Bearer {KEY}"}
PROJECT, USER = f"multiagent-{uuid.uuid4().hex[:6]}", "u1"
SUPPORT, SALES = "support-agent", "sales-agent"
c = httpx.Client(base_url=API, headers=H, timeout=60)


def ingest(agent, text):
    c.post("/v1/ingest", json={"project_id": PROJECT, "agent_id": agent, "user_id": USER,
           "messages": [{"role": "user", "content": text}], "response_text": "Noted."}).raise_for_status()


def memories(agent):
    return c.get("/v1/memory", params={"project_id": PROJECT, "agent_id": agent,
                 "user_id": USER, "limit": 50}).json()["memories"]


def search(agent, query):
    return c.get("/v1/memory/search", params={"project_id": PROJECT, "agent_id": agent,
                 "user_id": USER, "query": query, "limit": 5}).json()["memories"]


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY.")
    print(f"project={PROJECT} user={USER}\n")

    print(f"[{SUPPORT}] learns two things from the user...")
    ingest(SUPPORT, "Heads up: I'm allergic to penicillin.")
    ingest(SUPPORT, "By the way my favourite colour is teal.")
    time.sleep(8)

    mem = memories(SUPPORT)
    print(f"  {SUPPORT} now has {len(mem)} memories (all private by default):")
    for m in mem:
        print(f"    - [{m['scope']}] {m['content']}")

    # Share the safety-critical one with all agents; keep the chit-chat private.
    allergy = next((m for m in mem if "penicillin" in m["content"].lower()), None)
    if allergy:
        c.patch(f"/v1/memory/{allergy['id']}/scope", json={"scope": "global"}).raise_for_status()
        print(f"\n  -> promoted '{allergy['content']}' to GLOBAL (shared with all agents)")

    print(f"\n[{SALES}] — a DIFFERENT agent — looks the user up:")
    health = search(SALES, "any health issues or allergies I should know about?")
    colour = search(SALES, "what is the user's favourite colour?")
    print(f"  health query  -> {[(m['content'], 'shared from '+m['source_agent']) for m in health]}")
    print(f"  colour query  -> {[m['content'] for m in colour]}")

    saw_allergy = any("penicillin" in m["content"].lower() for m in health)
    saw_colour = any("teal" in m["content"].lower() for m in colour)
    print()
    print(f"  sales-agent sees the SHARED allergy : {saw_allergy}  (expected True)")
    print(f"  sales-agent sees the PRIVATE colour : {saw_colour}  (expected False)")
    print("MULTI-AGENT DEMO " + ("PASSED [OK]" if saw_allergy and not saw_colour else "CHECK OUTPUT [!]"))


if __name__ == "__main__":
    main()
