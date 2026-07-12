"""Wrap-side support for the NATIVE Anthropic SDK.

Anthropic's surface differs from OpenAI's: `client.messages.create(...)` with a
top-level `system` parameter instead of system-role messages. So enrichment
lands differently — the instruction + memory blocks are PREPENDED to `system`
(instructions first, then the developer's own system text, then recalled
memory), and the post-hook reads assistant text from `response.content`.

Same guarantees as the OpenAI proxy: identical response object back, ids
stripped before the call, fire-and-forget ingest, memory never breaks the call.
"""
import asyncio
import inspect
import logging
import threading

from memgram.sdk.assembler import (_format_instructions, _format_semantic,
                                   _last_user_text)
from memgram.sdk.proxy import _Passthrough

logger = logging.getLogger("memgram")


def _anthropic_text(response) -> str | None:
    try:
        return "".join(getattr(b, "text", "") for b in response.content) or None
    except Exception:
        return None


class MessagesProxy(_Passthrough):
    def __init__(self, inner, api, config):
        super().__init__(inner)
        self._api = api
        self._config = config
        self._is_async = inspect.iscoroutinefunction(inner.create)

    def create(self, **kwargs):
        return self._acreate(**kwargs) if self._is_async else self._create(**kwargs)

    # -- enrichment ----------------------------------------------------------
    def _blocks(self, user_id, agent_id, messages):
        instr = sem = None
        try:
            instr = _format_instructions(self._api.get_instructions(user_id, agent_id))
        except Exception as e:
            logger.warning("memgram: instruction fetch failed (%s); passing through", e)
        if self._config is None or self._config.features.get("semantic", True):
            try:
                q = _last_user_text(messages)
                if q:
                    sem = _format_semantic(self._api.search_memories(user_id, agent_id, q))
            except Exception as e:
                logger.warning("memgram: semantic fetch failed (%s); skipping", e)
        return instr, sem

    @staticmethod
    def _merge_system(existing, instr, sem):
        """Injection order preserved: instructions -> dev system -> memories.
        Anthropic allows `system` as str or content-block list; normalize to str."""
        if isinstance(existing, list):  # content blocks -> text
            existing = "\n".join(b.get("text", "") if isinstance(b, dict)
                                 else getattr(b, "text", "") for b in existing)
        parts = [p for p in (instr, existing, sem) if p]
        return "\n\n".join(parts) if parts else None

    def _pop_ids(self, kwargs):
        return (kwargs.pop("user_id", "default"),
                kwargs.pop("agent_id", self._config.agent_name))

    # -- sync path -------------------------------------------------------------
    def _create(self, **kwargs):
        user_id, agent_id = self._pop_ids(kwargs)
        original = list(kwargs.get("messages", []))
        instr, sem = self._blocks(user_id, agent_id, original)
        merged = self._merge_system(kwargs.get("system"), instr, sem)
        if merged is not None:
            kwargs["system"] = merged
        response = self._inner.create(**kwargs)
        text = _anthropic_text(response)
        threading.Thread(
            target=lambda: self._api.ingest(user_id, agent_id, original, text),
            daemon=True).start()
        return response

    # -- async path --------------------------------------------------------------
    async def _acreate(self, **kwargs):
        user_id, agent_id = self._pop_ids(kwargs)
        original = list(kwargs.get("messages", []))
        instr, sem = await asyncio.to_thread(self._blocks, user_id, agent_id, original)
        merged = self._merge_system(kwargs.get("system"), instr, sem)
        if merged is not None:
            kwargs["system"] = merged
        response = await self._inner.create(**kwargs)
        asyncio.create_task(self._api.aingest(
            user_id, agent_id, original, _anthropic_text(response)))
        return response


class AnthropicWrappedClient(_Passthrough):
    """Handed back by mem.wrap(anthropic.Anthropic()). Identical surface;
    only messages.create is enriched."""

    def __init__(self, inner, api, config):
        super().__init__(inner)
        self.messages = MessagesProxy(inner.messages, api, config)
