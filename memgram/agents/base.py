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
    """Cheap, deterministic work: extraction, summarization.
    Resolution: MEMGRAM_FAST_MODEL > the MEMGRAM_BRAIN preset's fast tier."""
    from memgram.llm import brain_spec
    return os.environ.get("MEMGRAM_FAST_MODEL") or brain_spec()["fast"]


def quality_model() -> str:
    """User-facing / long-term-shaping work: reflection, proposals.
    Resolution: MEMGRAM_QUALITY_MODEL > the MEMGRAM_BRAIN preset's quality tier."""
    return os.environ.get("MEMGRAM_QUALITY_MODEL") or __import__(
        "memgram.llm", fromlist=["brain_spec"]).brain_spec()["quality"]


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
                await self._flush_usage(job)
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
        self._track_usage(getattr(r, "usage", None))
        return r.choices[0].message.content

    def _track_usage(self, u) -> None:
        tin, tout = 0, 0
        if u is not None:
            from memgram.usage import norm_usage
            tin, tout = norm_usage(u)
        self._usage_in = getattr(self, "_usage_in", 0) + tin
        self._usage_out = getattr(self, "_usage_out", 0) + tout

    async def _flush_usage(self, job: dict) -> None:
        """Write accumulated token usage for this job. Best-effort."""
        tin, tout = getattr(self, "_usage_in", 0), getattr(self, "_usage_out", 0)
        self._usage_in = self._usage_out = 0
        pool = getattr(self.store, "pool", None)
        if pool is None or (tin == 0 and tout == 0) or "project_id" not in job:
            return
        from memgram.usage import record
        await record(pool, job["project_id"], job.get("agent_id", "?"),
                     job.get("user_id", "?"), f"llm:{type(self).__name__}",
                     model=self.model, tokens_in=tin, tokens_out=tout)

    @staticmethod
    def parse_json(raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # tolerate providers without native JSON mode (fences / stray prose)
            from memgram.llm import extract_json
            return json.loads(extract_json(raw))

    async def on_success(self, job: dict, result: dict) -> None: ...
    async def on_failure(self, job: dict, err: Exception) -> None: ...
