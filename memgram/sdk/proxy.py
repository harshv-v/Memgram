"""The intercept. Wraps any OpenAI-compatible client via __getattr__ passthrough.
Only `chat.completions.create` is intercepted; everything else passes through
untouched. Works with both sync and async clients.

Hot path: enrich -> real LLM call (unchanged) -> return the exact same object.
The post-hook (episodic log + extract job) fires fire-and-forget and NEVER
blocks the response: a daemon thread on the sync path, a task on the async path.
"""
import asyncio
import inspect
import logging
import threading

from memgram.sdk.assembler import ContextAssembler
from memgram.sdk.config import MemgramConfig

logger = logging.getLogger("memgram")


def _response_text(response) -> str | None:
    """Best-effort pull of assistant text. Tolerates dicts, objects, and
    streaming (None for streams — nothing to log until accumulation lands)."""
    try:
        if isinstance(response, dict):
            return response.get("choices", [{}])[0].get("message", {}).get("content")
        return response.choices[0].message.content
    except Exception:
        return None


def _delta_text(chunk) -> str:
    """Text delta from a streaming chunk (dict or SDK object), else ''."""
    try:
        if isinstance(chunk, dict):
            return chunk.get("choices", [{}])[0].get("delta", {}).get("content") or ""
        return chunk.choices[0].delta.content or ""
    except Exception:
        return ""


class _SyncStream:
    """Wraps a sync streaming response: passes every chunk through untouched,
    accumulates the assistant text, and fires `on_done(text)` exactly once when
    the stream ends. The developer's iteration code doesn't change."""

    def __init__(self, inner, on_done):
        self._inner = inner
        self._on_done = on_done
        self._parts: list[str] = []
        self._fired = False

    def __iter__(self):
        try:
            for chunk in self._inner:
                self._parts.append(_delta_text(chunk))
                yield chunk
        finally:
            self._finish()

    def _finish(self):
        if not self._fired:
            self._fired = True
            self._on_done("".join(self._parts) or None)

    def __enter__(self):
        getattr(self._inner, "__enter__", lambda: None)()
        return self

    def __exit__(self, *exc):
        return getattr(self._inner, "__exit__", lambda *a: None)(*exc)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _AsyncStream:
    """Async twin of _SyncStream."""

    def __init__(self, inner, on_done):
        self._inner = inner
        self._on_done = on_done
        self._parts: list[str] = []
        self._fired = False

    async def __aiter__(self):
        try:
            async for chunk in self._inner:
                self._parts.append(_delta_text(chunk))
                yield chunk
        finally:
            await self._finish()

    async def _finish(self):
        if not self._fired:
            self._fired = True
            await self._on_done("".join(self._parts) or None)

    async def __aenter__(self):
        aenter = getattr(self._inner, "__aenter__", None)
        if aenter:
            await aenter()
        return self

    async def __aexit__(self, *exc):
        aexit = getattr(self._inner, "__aexit__", None)
        return await aexit(*exc) if aexit else None

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Passthrough:
    """Delegates every attribute to the wrapped object."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)


class ChatCompletionsProxy(_Passthrough):
    def __init__(self, inner, assembler: ContextAssembler, config: MemgramConfig):
        super().__init__(inner)
        self._assembler = assembler
        self._api = assembler._api
        self._config = config
        self._is_async = inspect.iscoroutinefunction(inner.create)

    def create(self, **kwargs):
        if self._is_async:
            return self._acreate(**kwargs)
        return self._create(**kwargs)

    # -- sync path ----------------------------------------------------------
    def _create(self, **kwargs):
        user_id, agent_id = self._pop_ids(kwargs)
        original = list(kwargs["messages"])
        kwargs["messages"] = self._assembler.enrich(
            original, user_id=user_id, agent_id=agent_id
        )
        response = self._inner.create(**kwargs)
        if kwargs.get("stream"):
            # v1.1 gap closed: accumulate chunks, post-hook on completion.
            def _done(text):
                def _run():
                    self._api.ingest(user_id, agent_id, original, text)
                threading.Thread(target=_run, daemon=True).start()
            return _SyncStream(response, _done)
        self._post_sync(original, response, user_id, agent_id)
        return response

    # -- async path -----------------------------------------------------------
    async def _acreate(self, **kwargs):
        user_id, agent_id = self._pop_ids(kwargs)
        original = list(kwargs["messages"])
        kwargs["messages"] = await self._assembler.aenrich(
            original, user_id=user_id, agent_id=agent_id
        )
        response = await self._inner.create(**kwargs)
        if kwargs.get("stream"):
            async def _done(text):
                try:
                    await self._api.aingest(user_id, agent_id, original, text)
                except Exception as e:
                    logger.debug("memgram stream post-hook failed (ignored): %s", e)
            return _AsyncStream(response, _done)
        # Never blocks the response — schedule and return immediately.
        asyncio.create_task(self._post_async(original, response, user_id, agent_id))
        return response

    def _pop_ids(self, kwargs) -> tuple[str, str]:
        user_id = kwargs.pop("user_id", "default")
        agent_id = kwargs.pop("agent_id", self._config.agent_name)
        return user_id, agent_id

    # -- post-hooks (fire-and-forget) --------------------------------------
    def _post_sync(self, messages, response, user_id, agent_id) -> None:
        text = _response_text(response)

        def _run():
            self._api.ingest(user_id, agent_id, messages, text)
        threading.Thread(target=_run, daemon=True).start()

    async def _post_async(self, messages, response, user_id, agent_id) -> None:
        try:
            await self._api.aingest(user_id, agent_id, messages,
                                    _response_text(response))
        except Exception as e:
            logger.debug("memgram post-hook failed (ignored): %s", e)


class ChatProxy(_Passthrough):
    def __init__(self, inner, assembler, config):
        super().__init__(inner)
        self.completions = ChatCompletionsProxy(inner.completions, assembler, config)


class WrappedClient(_Passthrough):
    """The object handed back to the developer. Identical surface to the
    original client; only chat.completions.create is enriched."""

    def __init__(self, inner, assembler: ContextAssembler, config: MemgramConfig):
        super().__init__(inner)
        self.chat = ChatProxy(inner.chat, assembler, config)
