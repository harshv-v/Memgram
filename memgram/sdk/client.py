"""HTTP client to the Memgram backend API. Sync + async variants — the proxy
uses whichever matches the wrapped LLM client."""
import httpx

from memgram.sdk.config import MemgramConfig


class MemgramAPIClient:
    def __init__(self, config: MemgramConfig):
        self._config = config
        self._headers = {"Authorization": f"Bearer {config.api_key}"}
        self._sync = httpx.Client(
            base_url=config.api_base_url, headers=self._headers, timeout=config.timeout
        )
        self._async = httpx.AsyncClient(
            base_url=config.api_base_url, headers=self._headers, timeout=config.timeout
        )

    def _params(self, user_id: str, agent_id: str) -> dict:
        return {
            "project_id": self._config.project_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "status": "active",
        }

    # -- instructions: read (hot path) ------------------------------------
    def get_instructions(self, user_id: str, agent_id: str) -> list[dict]:
        r = self._sync.get("/v1/instructions", params=self._params(user_id, agent_id))
        r.raise_for_status()
        return r.json()["instructions"]

    async def aget_instructions(self, user_id: str, agent_id: str) -> list[dict]:
        r = await self._async.get("/v1/instructions", params=self._params(user_id, agent_id))
        r.raise_for_status()
        return r.json()["instructions"]

    # -- semantic memory: read (hot path) ----------------------------------
    def search_memories(self, user_id: str, agent_id: str, query: str,
                        limit: int = 5) -> list[dict]:
        r = self._sync.get("/v1/memory/search", params={
            "project_id": self._config.project_id, "agent_id": agent_id,
            "user_id": user_id, "query": query, "limit": limit,
        })
        r.raise_for_status()
        return r.json()["memories"]

    async def asearch_memories(self, user_id: str, agent_id: str, query: str,
                              limit: int = 5) -> list[dict]:
        r = await self._async.get("/v1/memory/search", params={
            "project_id": self._config.project_id, "agent_id": agent_id,
            "user_id": user_id, "query": query, "limit": limit,
        })
        r.raise_for_status()
        return r.json()["memories"]

    # -- ingest: the async post-hook target (fire-and-forget) --------------
    def ingest(self, user_id: str, agent_id: str, messages: list[dict],
               response_text: str | None = None) -> None:
        try:
            self._sync.post("/v1/ingest", json={
                "project_id": self._config.project_id, "agent_id": agent_id,
                "user_id": user_id, "messages": _clean(messages),
                "response_text": response_text,
            })
        except Exception:
            pass  # memory ingestion must never surface to the developer

    async def aingest(self, user_id: str, agent_id: str, messages: list[dict],
                      response_text: str | None = None) -> None:
        try:
            await self._async.post("/v1/ingest", json={
                "project_id": self._config.project_id, "agent_id": agent_id,
                "user_id": user_id, "messages": _clean(messages),
                "response_text": response_text,
            })
        except Exception:
            pass

    # -- instructions: write (used by apps / the demo) ---------------------
    def create_instruction(
        self, user_id: str, agent_id: str, content: str,
        priority: int = 2, source: str = "user",
    ) -> dict:
        r = self._sync.post("/v1/instructions", json={
            "project_id": self._config.project_id, "agent_id": agent_id,
            "user_id": user_id, "content": content,
            "priority": priority, "source": source,
        })
        r.raise_for_status()
        return r.json()


def _clean(messages: list[dict]) -> list[dict]:
    """Strip injected memory system-blocks before logging — we only persist the
    developer's own turns, never our own injected context (avoids feedback loops)."""
    out = []
    for m in messages:
        c = m.get("content")
        if not isinstance(c, str):
            continue
        if m.get("role") == "system" and (
            c.startswith("## User memory") or c.startswith("## Relevant memory")
        ):
            continue
        out.append({"role": m["role"], "content": c})
    return out
