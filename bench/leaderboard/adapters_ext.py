"""Leaderboard adapters beyond the core two (Memgram, Mem0 live in
bench/quality/adapters.py). Same surface: setup / ingest / wait_ready / retrieve.

Each competitor SDK is imported lazily and optionally — a missing or broken
SDK SKIPS that system with a message instead of failing the board. Zep and
Letta are SERVICES: they need an API key (cloud) or a running server, and
their SDK surfaces move fast — each adapter notes the SDK version it targets.
Verify against current docs before publishing numbers.
"""
import time
import uuid


class NoMemoryBaseline:
    """The control: no memory layer at all — every past turn is re-stuffed into
    context. Any memory system must beat this on tokens, and should approach it
    on recall (it 'remembers' everything, at maximum token cost)."""
    name = "no-memory"

    def __init__(self):
        self.turns: dict[str, list[str]] = {}

    def setup(self, persona):
        self.turns[persona] = []

    def ingest(self, persona, turns):
        self.turns.setdefault(persona, []).extend(turns)

    def wait_ready(self):
        pass

    def retrieve(self, persona, query, k=6):
        return list(self.turns.get(persona, []))  # everything, always


class ZepAdapter:
    """Zep (temporal knowledge graph). Targets zep-cloud>=2.x — needs
    ZEP_API_KEY (cloud) or a self-hosted server. VERIFY against current Zep
    docs: the graph/memory API surface has changed between majors."""
    name = "zep"

    def __init__(self):
        import os
        from zep_cloud.client import Zep
        self.client = Zep(api_key=os.environ["ZEP_API_KEY"])
        self.run_id = uuid.uuid4().hex[:6]

    def _uid(self, persona):
        return f"board-{self.run_id}-{persona}"

    def setup(self, persona):
        try:
            self.client.user.add(user_id=self._uid(persona))
        except Exception:
            pass  # already exists

    def ingest(self, persona, turns):
        from zep_cloud.types import Message
        session_id = f"{self._uid(persona)}-{uuid.uuid4().hex[:6]}"
        self.client.memory.add_session(session_id=session_id, user_id=self._uid(persona))
        self.client.memory.add(session_id=session_id, messages=[
            Message(role="user", role_type="user", content=t) for t in turns])

    def wait_ready(self):
        time.sleep(20)  # graph construction is async server-side; generous grace

    def retrieve(self, persona, query, k=6):
        r = self.client.graph.search(user_id=self._uid(persona), query=query, limit=k)
        out = []
        for e in (getattr(r, "edges", None) or []):
            fact = getattr(e, "fact", None)
            if fact:
                out.append(fact)
        for n in (getattr(r, "nodes", None) or []):
            summ = getattr(n, "summary", None)
            if summ:
                out.append(summ)
        return out[:k]


class LettaAdapter:
    """Letta (MemGPT). Targets letta-client>=0.1 against a running Letta server
    (docker run lettaai/letta). One agent per persona; recall via archival
    passage search. VERIFY against current Letta docs."""
    name = "letta"

    def __init__(self):
        import os
        from letta_client import Letta
        self.client = Letta(base_url=os.environ.get("LETTA_BASE_URL", "http://localhost:8283"))
        self.agents: dict[str, str] = {}

    def setup(self, persona):
        agent = self.client.agents.create(
            name=f"board-{persona}-{uuid.uuid4().hex[:6]}",
            memory_blocks=[{"label": "human", "value": ""},
                           {"label": "persona", "value": "You remember facts about the user."}],
            model="openai/gpt-4o-mini", embedding="openai/text-embedding-3-small")
        self.agents[persona] = agent.id

    def ingest(self, persona, turns):
        for t in turns:
            self.client.agents.messages.create(
                agent_id=self.agents[persona],
                messages=[{"role": "user", "content": t}])

    def wait_ready(self):
        pass  # Letta processes synchronously per message

    def retrieve(self, persona, query, k=6):
        out = []
        # archival passages + core memory both count as "what it remembered"
        try:
            passages = self.client.agents.passages.list(
                agent_id=self.agents[persona], search=query, limit=k)
            out += [getattr(p, "text", "") for p in passages]
        except Exception:
            pass
        try:
            core = self.client.agents.core_memory.retrieve(agent_id=self.agents[persona])
            for block in getattr(core, "blocks", []) or []:
                v = getattr(block, "value", "")
                if v:
                    out.append(v)
        except Exception:
            pass
        return [o for o in out if o][:k]


class LangMemAdapter:
    """LangMem (LangChain's memory SDK) over an in-memory LangGraph store.
    Targets langmem>=0.0.x. VERIFY against current LangMem docs."""
    name = "langmem"

    def __init__(self):
        from langgraph.store.memory import InMemoryStore
        from langmem import create_memory_store_manager
        self.store = InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})
        self.manager = create_memory_store_manager(
            "openai:gpt-4o-mini", namespace=("memories", "{user_id}"), store=self.store)

    def setup(self, persona):
        pass

    def ingest(self, persona, turns):
        msgs = {"messages": [{"role": "user", "content": t} for t in turns]}
        self.manager.invoke(msgs, config={"configurable": {"user_id": persona}})

    def wait_ready(self):
        pass

    def retrieve(self, persona, query, k=6):
        items = self.store.search(("memories", persona), query=query, limit=k)
        out = []
        for it in items:
            v = it.value
            out.append(v.get("content", str(v)) if isinstance(v, dict) else str(v))
        return out


EXT_ADAPTERS = {
    "no-memory": NoMemoryBaseline,
    "zep": ZepAdapter,
    "letta": LettaAdapter,
    "langmem": LangMemAdapter,
}
