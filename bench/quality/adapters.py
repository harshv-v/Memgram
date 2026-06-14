"""Memory-system adapters for the quality eval. Each exposes the same surface so
the harness can score them identically:

    setup(persona)            -> fresh, isolated memory space for one persona
    ingest(persona, turns)    -> store one session's user turns
    wait_ready()              -> block until async extraction has settled
    retrieve(persona, q, k)   -> return up to k memory strings relevant to q

Only retrieval quality differs between systems — answer generation and grading
are shared (see run_eval.py), so the comparison is apples-to-apples.
"""
import os
import time
import uuid

import httpx


class MemgramAdapter:
    name = "memgram"

    def __init__(self):
        self.base = os.environ.get("MEMGRAM_API_BASE_URL", "http://localhost:8000")
        self.key = os.environ.get("MEMGRAM_API_KEY", "mgram_dev_key")
        self.c = httpx.Client(base_url=self.base,
                              headers={"Authorization": f"Bearer {self.key}"}, timeout=60)
        self.project = f"eval-{uuid.uuid4().hex[:6]}"
        self.agent = "eval-agent"

    def setup(self, persona):  # one project, persona as user_id -> isolated by RLS
        pass

    def ingest(self, persona, turns):
        msgs = [{"role": "user", "content": t} for t in turns]
        self.c.post("/v1/ingest", json={
            "project_id": self.project, "agent_id": self.agent, "user_id": persona,
            "messages": msgs, "response_text": "Understood."}).raise_for_status()

    def wait_ready(self):
        # async extraction: wait for the queue to drain, then a short grace
        deadline = time.time() + 90
        while time.time() < deadline:
            h = self.c.get("/health").json()
            if h.get("queue_depth", 1) == 0:
                break
            time.sleep(0.5)
        time.sleep(1.5)

    def retrieve(self, persona, query, k=6):
        r = self.c.get("/v1/memory/search", params={
            "project_id": self.project, "agent_id": self.agent, "user_id": persona,
            "query": query, "limit": k})
        r.raise_for_status()
        return [m["content"] for m in r.json()["memories"]]


class Mem0Adapter:
    name = "mem0"

    def __init__(self):
        from mem0 import Memory
        path = f"/tmp/qdrant_eval_{uuid.uuid4().hex[:8]}"
        self.m = Memory.from_config({
            "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
            "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
            "vector_store": {"provider": "qdrant", "config": {"path": path, "on_disk": True}},
        })

    def setup(self, persona):
        pass

    def ingest(self, persona, turns):
        msgs = [{"role": "user", "content": t} for t in turns]
        self.m.add(msgs, user_id=persona)  # synchronous: extraction happens inline

    def wait_ready(self):
        pass  # sync

    def retrieve(self, persona, query, k=6):
        res = self.m.search(query, filters={"user_id": persona}, top_k=k)
        items = res.get("results") if isinstance(res, dict) else res
        return [r["memory"] for r in items]


ADAPTERS = {"memgram": MemgramAdapter, "mem0": Mem0Adapter}
