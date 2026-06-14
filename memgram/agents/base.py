"""BaseAgent — the whole agent framework, ~60 lines, zero dependencies beyond
the OpenAI client. No LangChain, no LangGraph. An agent is: a prompt, a JSON
response, and a retry loop. That's it.

The LLM client is injected so any OpenAI-compatible endpoint works (Ollama,
vLLM, Groq) and tests run with a fake.
"""
import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger("memgram.agents")


def fast_model() -> str:
    """Cheap, deterministic work: extraction, summarization."""
    return os.environ.get("MEMGRAM_FAST_MODEL", "gpt-4o-mini")


def quality_model() -> str:
    """User-facing / long-term-shaping work: reflection, proposals.
    Design specifies gpt-4o here; override via MEMGRAM_QUALITY_MODEL (e.g. if
    gpt-4o is retired, point this at the current higher tier)."""
    return os.environ.get("MEMGRAM_QUALITY_MODEL", "gpt-4o")


class BaseAgent(ABC):
    model = "gpt-4o-mini"  # override per agent; env-overridable at worker level
    max_retries = 3

    def __init__(self, store, config: dict | None = None, llm=None):
        self.store = store
        self.config = config or {}
        self._llm = llm  # AsyncOpenAI-compatible; None for SQL-only agents
        if m := self.config.get("model"):
            self.model = m

    @abstractmethod
    def build_prompt(self, job: dict) -> list[dict]: ...

    @abstractmethod
    def parse_output(self, raw: str) -> dict: ...

    async def run(self, job: dict) -> dict | None:
        for attempt in range(self.max_retries):
            try:
                messages = self.build_prompt(job)
                raw = await self._call_llm(messages)
                result = self.parse_output(raw)
                await self.on_success(job, result)
                return result
            except Exception as e:
                logger.warning("%s attempt %d failed: %s",
                               type(self).__name__, attempt + 1, e)
                if attempt == self.max_retries - 1:
                    await self.on_failure(job, e)
                    return None
                await asyncio.sleep(2 ** attempt)

    async def _call_llm(self, messages: list[dict]) -> str:
        r = await self._llm.chat.completions.create(
            model=self.model, messages=messages,
            response_format={"type": "json_object"},
        )
        return r.choices[0].message.content

    @staticmethod
    def parse_json(raw: str) -> dict:
        return json.loads(raw)

    async def on_success(self, job: dict, result: dict) -> None: ...
    async def on_failure(self, job: dict, err: Exception) -> None: ...
