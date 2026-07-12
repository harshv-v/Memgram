"""Worker entrypoint:  python -m memgram.worker"""
import asyncio
import os

import asyncpg

from memgram.obs import setup_logging

from memgram.memory.embedder import get_embedder
from memgram.memory.store import MemoryStore
from memgram.presets import resolve
from memgram.worker.dispatcher import Dispatcher
from memgram.worker.queue import JobQueue
from memgram.worker.scheduler import Scheduler

setup_logging()


def worker_config() -> dict:
    """Resolve the same preset the SDK uses (MEMGRAM_PRESET) into the flat config
    the dispatcher/scheduler read, so feature flags and tuning match end to end."""
    s = resolve(os.environ.get("MEMGRAM_PRESET"))
    return {
        "features": s["features"],
        "reflect_every_n": s["reflection"]["trigger_every_n"],
        "reflection": {"habit_threshold": s["decay"]["habit_threshold"]},
        "decay": {"demotion_days": s["decay"]["demotion_days"]},
        "reflect_every_hrs": s["reflection"]["trigger_every_hrs"],
        "concurrency": int(os.environ.get("MEMGRAM_WORKER_CONCURRENCY", "8")),
    }


def _use_uvloop():
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass


async def main() -> None:
    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=1,
        max_size=int(os.environ.get("MEMGRAM_POOL_MAX", "5")))
    embedder = get_embedder()
    store = MemoryStore(pool, embedder)
    queue = JobQueue()
    cfg = worker_config()
    dispatcher = Dispatcher(store, queue, embedder, cfg)
    scheduler = Scheduler(pool, queue, cfg)
    await asyncio.gather(dispatcher.run_forever(), scheduler.run_forever())


if __name__ == "__main__":
    _use_uvloop()
    asyncio.run(main())
