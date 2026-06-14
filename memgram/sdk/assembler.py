"""Context assembler — builds the enriched prompt in strict injection-priority
order (section 4 of the design doc):

  1. User instructions       — full store, FIRST, never truncated.
  2. Developer system prompt  — unchanged, left where the developer put it.
  3. Semantic memories        — top-k, injected right AFTER the dev system prompt.
  ... (episodic / procedural arrive with their pillars)

Respects feature flags (semantic can be turned off via preset). Memory must
NEVER break the LLM call: every fetch is wrapped; any failure falls through to
the original messages untouched.
"""
import logging
import os

from memgram.sdk.client import MemgramAPIClient

logger = logging.getLogger("memgram")

_INSTR_HEADER = (
    "## User memory — standing instructions\n"
    "These are persistent preferences this user has set. Follow them in every "
    "response, in priority order (1 = always, 3 = soft preference):\n"
)
_SEM_HEADER = (
    "## Relevant memory — what you already know about this user\n"
    "Recalled from earlier sessions. Use it silently; do not announce that you "
    "remembered:\n"
)


def _format_instructions(instructions: list[dict]) -> str | None:
    if not instructions:
        return None
    lines = [f"- [priority {i['priority']}] {i['content']}" for i in instructions]
    return _INSTR_HEADER + "\n".join(lines)


def _format_semantic(memories: list[dict]) -> str | None:
    if not memories:
        return None
    lines = [f"- {m['content']}" for m in memories]
    return _SEM_HEADER + "\n".join(lines)


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return ""


def _inject(messages: list[dict], instr_block: str | None,
            sem_block: str | None) -> list[dict]:
    if instr_block is None and sem_block is None:
        return messages
    msgs = list(messages)
    result: list[dict] = []
    if instr_block is not None:                         # 1. instructions, first
        result.append({"role": "system", "content": instr_block})
    rest = msgs
    if msgs and msgs[0].get("role") == "system":        # 2. dev system prompt, kept
        result.append(msgs[0])
        rest = msgs[1:]
    if sem_block is not None:                           # 3. semantic, after dev system
        result.append({"role": "system", "content": sem_block})
    result.extend(rest)                                 # ...the rest of the conversation
    if os.environ.get("MEMGRAM_DEBUG"):
        logger.warning("memgram injected blocks: instr=%s sem=%s",
                       instr_block is not None, sem_block is not None)
    return result


class ContextAssembler:
    def __init__(self, api: MemgramAPIClient, config=None):
        self._api = api
        self._config = config

    def _semantic_on(self) -> bool:
        return self._config is None or self._config.features.get("semantic", True)

    def _budget(self) -> int:
        return self._config.memory_budget if self._config else 4000

    def enrich(self, messages: list[dict], user_id: str, agent_id: str) -> list[dict]:
        instr_block = sem_block = None
        try:
            instr_block = _format_instructions(self._api.get_instructions(user_id, agent_id))
        except Exception as e:
            logger.warning("memgram: instruction fetch failed (%s); passing through", e)
        if self._semantic_on():
            try:
                query = _last_user_text(messages)
                if query:
                    sem_block = _format_semantic(self._api.search_memories(user_id, agent_id, query))
            except Exception as e:
                logger.warning("memgram: semantic fetch failed (%s); skipping", e)
        return _inject(messages, instr_block, sem_block)

    async def aenrich(self, messages: list[dict], user_id: str, agent_id: str) -> list[dict]:
        instr_block = sem_block = None
        try:
            instr_block = _format_instructions(await self._api.aget_instructions(user_id, agent_id))
        except Exception as e:
            logger.warning("memgram: instruction fetch failed (%s); passing through", e)
        if self._semantic_on():
            try:
                query = _last_user_text(messages)
                if query:
                    sem_block = _format_semantic(await self._api.asearch_memories(user_id, agent_id, query))
            except Exception as e:
                logger.warning("memgram: semantic fetch failed (%s); skipping", e)
        return _inject(messages, instr_block, sem_block)
