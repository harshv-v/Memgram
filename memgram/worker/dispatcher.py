"""Dispatcher — routes queued jobs to agents. The whole 'orchestration
framework' is this dict and a loop."""
import asyncio
import logging
import os

from memgram.agents.decay import DecayAgent
from memgram.agents.extractor import ExtractorAgent
from memgram.agents.proposer import ProposerAgent
from memgram.agents.reflection import ReflectionAgent
from memgram.agents.summarizer import SummarizerAgent

logger = logging.getLogger("memgram.worker")


def get_llm():
    """Any OpenAI-compatible endpoint. MEMGRAM_FAKE_LLM=1 → canned responses
    (full pipeline runs with zero external calls)."""
    if os.environ.get("MEMGRAM_FAKE_LLM"):
        from memgram.testing import FakeLLM
        return FakeLLM()
    from openai import AsyncOpenAI
    # `or None`: compose injects MEMGRAM_LLM_BASE_URL="" by default, and an empty
    # string is NOT the same as unset — it would override the OpenAI default with
    # a blank base URL and every request would fail with "Connection error".
    base_url = os.environ.get("MEMGRAM_LLM_BASE_URL") or None
    return AsyncOpenAI(base_url=base_url)


class Dispatcher:
    def __init__(self, store, queue, embedder, config: dict | None = None):
        llm = get_llm()
        config = config or {}
        self.queue = queue
        self.agents = {
            "extract":   ExtractorAgent(store, config.get("extractor"), llm),
            "summarize": SummarizerAgent(store, config.get("summarizer"), llm),
            "reflect":   ReflectionAgent(store, config.get("reflection"), llm, queue=queue),
            "propose":   ProposerAgent(store, config.get("proposer"), llm, embedder=embedder),
            "decay":     DecayAgent(store, config.get("decay")),
        }
        # extract jobs count toward the reflection cadence
        self.reflect_every = int(config.get("reflect_every_n", 20))
        self.features = config.get("features", {})

    async def handle(self, job: dict) -> None:
        agent = self.agents.get(job["type"])
        if agent is None:
            logger.error("unknown job type %s", job["type"])
            return
        result = await agent.run(job["payload"])
        logger.info("job %s (%s) -> %s", job["id"], job["type"], result)

        if job["type"] == "extract" and self.features.get("reflection", True):
            p = job["payload"]
            n = self.queue.bump_interaction(p["project_id"], p["user_id"], p["agent_id"])
            if n % self.reflect_every == 0:
                self.queue.enqueue("reflect", {
                    "project_id": p["project_id"], "user_id": p["user_id"],
                    "agent_id": p["agent_id"],
                })

    async def run_forever(self, poll_timeout: int = 5) -> None:
        logger.info("memgram worker started")
        while True:
            try:
                job = await asyncio.to_thread(self.queue.dequeue, poll_timeout)
            except Exception:
                # A transient queue read failure must never kill the worker.
                logger.exception("dequeue failed; backing off")
                await asyncio.sleep(1)
                continue
            if job is None:
                continue
            try:
                await self.handle(job)
            except Exception:
                logger.exception("job %s crashed", job.get("id"))

    async def drain(self) -> int:
        """Process everything currently queued, then return. Used by tests."""
        n = 0
        while True:
            job = self.queue.dequeue(timeout=1)
            if job is None:
                return n
            await self.handle(job)
            n += 1
