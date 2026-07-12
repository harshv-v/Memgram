"""Dispatcher — routes queued jobs to agents. The whole 'orchestration
framework' is this dict and a loop."""
import asyncio
import logging
import os
import time

from memgram.obs import JOBS_TOTAL, JOB_SECONDS
from memgram.agents.decay import DecayAgent
from memgram.agents.monitors import MonitorSuite
from memgram.agents.extractor import ExtractorAgent
from memgram.agents.procedural import ProceduralAgent
from memgram.agents.proposer import ProposerAgent
from memgram.agents.reflection import ReflectionAgent
from memgram.agents.summarizer import SummarizerAgent

logger = logging.getLogger("memgram.worker")


def get_llm():
    """The worker's brain. MEMGRAM_FAKE_LLM=1 -> canned responses (zero external
    calls); otherwise MEMGRAM_BRAIN picks the provider (openai | deepseek |
    gemini | groq | anthropic | watsonx) — see memgram/llm.py."""
    if os.environ.get("MEMGRAM_FAKE_LLM"):
        from memgram.testing import FakeLLM
        return FakeLLM()
    from memgram.llm import get_brain_llm
    return get_brain_llm()



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
            "monitor":   MonitorSuite(store, config.get("monitors")),
            "procedural": ProceduralAgent(store, config.get("procedural"), llm),
        }
        # extract jobs count toward the reflection cadence
        self.reflect_every = int(config.get("reflect_every_n", 20))
        self.features = config.get("features", {})
        # durability tuning
        self.reclaim_every = float(config.get("reclaim_every_s", 30))
        self.reclaim_idle_ms = int(config.get("reclaim_idle_ms", 60_000))
        self.max_deliveries = int(config.get("max_deliveries", 5))
        # how many jobs to process at once (consumer-group members in one process)
        self.concurrency = max(1, int(config.get("concurrency", 8)))

    async def handle(self, job: dict):
        """Run the job's agent. Returns the agent result; None means the agent
        exhausted its own retries (or the type is unknown). Raises only on a
        genuinely unexpected error, which the caller treats as 'leave pending'."""
        agent = self.agents.get(job["type"])
        if agent is None:
            logger.error("unknown job type %s", job["type"])
            return None
        t0 = time.perf_counter()
        result = await agent.run(job["payload"])
        JOB_SECONDS.labels(type=job["type"]).observe(time.perf_counter() - t0)
        JOBS_TOTAL.labels(type=job["type"],
                          status="ok" if result is not None else "failed").inc()
        logger.info("job %s (%s) -> %s", job["id"], job["type"], result)

        if job["type"] == "extract" and self.features.get("reflection", True):
            p = job["payload"]
            n = self.queue.bump_interaction(p["project_id"], p["user_id"], p["agent_id"])
            if n % self.reflect_every == 0:
                self.queue.enqueue("reflect", {
                    "project_id": p["project_id"], "user_id": p["user_id"],
                    "agent_id": p["agent_id"],
                })

        # If tools were used and procedural memory is on, learn from them.
        if job["type"] == "extract" and self.features.get("procedural", False):
            p = job["payload"]
            if any(m.get("role") in ("tool_call", "tool_result")
                   for m in p.get("messages", [])):
                self.queue.enqueue("procedural", p)
        return result

    async def _process(self, job: dict) -> None:
        # Too many deliveries (it keeps crashing the worker) -> quarantine it.
        if self.queue.deliveries(job) > self.max_deliveries:
            logger.error("job %s exceeded %d deliveries -> dead-letter",
                         job.get("id"), self.max_deliveries)
            self.queue.dead_letter(job, "max deliveries exceeded")
            return
        try:
            result = await self.handle(job)
        except Exception:
            # Leave UNACKED: it stays in the group's pending list and is
            # reclaimed + retried by a worker later. Nothing is lost on a crash.
            logger.exception("job %s crashed; left pending for redelivery", job.get("id"))
            return
        if result is None:
            logger.warning("job %s produced no result -> dead-letter", job.get("id"))
            self.queue.dead_letter(job, "no result after retries")
        else:
            self.queue.ack(job)

    async def _reclaim_stale(self) -> None:
        try:
            jobs = await asyncio.to_thread(self.queue.reclaim, self.reclaim_idle_ms)
        except Exception:
            logger.exception("reclaim failed")
            return
        for job in jobs:
            logger.warning("reclaimed stranded job %s (%s)", job.get("id"), job["type"])
            await self._process(job)

    async def run_forever(self, poll_timeout: int = 5) -> None:
        logger.info("memgram worker started (concurrency=%d)", self.concurrency)
        # N consumer loops process jobs in parallel (each is a member of the same
        # Streams group, so the stream load-balances across them) + 1 reclaimer.
        tasks = [asyncio.create_task(self._consume_loop(poll_timeout))
                 for _ in range(self.concurrency)]
        tasks.append(asyncio.create_task(self._reclaim_loop()))
        await asyncio.gather(*tasks)

    async def _consume_loop(self, poll_timeout: int) -> None:
        while True:
            try:
                job = await asyncio.to_thread(self.queue.dequeue, poll_timeout)
            except Exception:
                # A transient queue read failure must never kill the worker.
                logger.exception("dequeue failed; backing off")
                await asyncio.sleep(1)
                continue
            if job is not None:
                await self._process(job)

    async def _reclaim_loop(self) -> None:
        while True:
            await asyncio.sleep(self.reclaim_every)
            await self._reclaim_stale()

    async def drain(self) -> int:
        """Process everything currently queued, then return. Used by tests."""
        n = 0
        while True:
            job = self.queue.dequeue(timeout=1)
            if job is None:
                return n
            await self.handle(job)
            self.queue.ack(job)
            n += 1
