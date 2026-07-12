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

    # -- combined context: ONE round trip (hot path, preferred) -------------
    def get_context(self, user_id: str, agent_id: str, query: str | None,
                    limit: int = 5) -> dict | None:
        """instructions + memories in one call. None -> server predates
        /v1/context; caller falls back to the two-call path."""
        params = {**self._params(user_id, agent_id), "limit": limit}
        params.pop("status", None)
        if query:
            params["query"] = query
        r = self._sync.get("/v1/context", params=params)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    async def aget_context(self, user_id: str, agent_id: str, query: str | None,
                           limit: int = 5) -> dict | None:
        params = {**self._params(user_id, agent_id), "limit": limit}
        params.pop("status", None)
        if query:
            params["query"] = query
        r = await self._async.get("/v1/context", params=params)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

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


def _attr(o, k):
    """Read a field from either a dict or an OpenAI SDK message/tool object."""
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def _clean(messages: list) -> list[dict]:
    """Normalize the turns we persist. Strips our own injected memory blocks
    (avoids feedback loops) and flattens tool usage into loggable text turns:
    an assistant tool call becomes a `tool_call` turn and a tool result becomes
    a `tool_result` turn — so procedural memory can learn from what tools did."""
    out = []
    for m in messages:
        role = _attr(m, "role")
        # assistant tool call: content is usually None — preserve the call itself
        tool_calls = _attr(m, "tool_calls")
        if role == "assistant" and tool_calls:
            calls = []
            for tc in tool_calls:
                fn = _attr(tc, "function") or {}
                name = _attr(fn, "name") or "tool"
                args = _attr(fn, "arguments") or ""
                calls.append(f"{name}({args})")
            out.append({"role": "tool_call", "content": "; ".join(calls)})
            continue
        c = _attr(m, "content")
        if role == "tool" and isinstance(c, str):  # tool result
            # Tool outputs can be huge (a full API payload). Store a digest, not
            # the raw blob — we only need enough to learn the success/failure shape.
            out.append({"role": "tool_result", "content": c[:800]})
            continue
        if not isinstance(c, str):
            continue
        if role == "system" and (
            c.startswith("## User memory") or c.startswith("## Relevant memory")
        ):
            continue
        out.append({"role": role, "content": c})
    return out
