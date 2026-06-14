"""Redis-backed job queue. ~80 lines, no Celery. Works against Redis OR
Valkey (we code to the protocol). Idempotency via SET NX with TTL.

Set MEMGRAM_FAKE_REDIS=1 to run on fakeredis (tests / no-infra dev).
"""
import json
import os
import uuid

try:
    from redis.exceptions import TimeoutError as RedisTimeoutError
except Exception:  # redis always installed, but never let an import kill the worker
    RedisTimeoutError = ()

QUEUE_KEY = "memgram:jobs"
DONE_PREFIX = "memgram:done:"
DONE_TTL = 7 * 86400


def get_redis():
    if os.environ.get("MEMGRAM_FAKE_REDIS"):
        import fakeredis
        # singleton server so API process & worker loop share state in tests
        global _FAKE_SERVER
        try:
            _FAKE_SERVER
        except NameError:
            _FAKE_SERVER = fakeredis.FakeServer()
        return fakeredis.FakeRedis(server=_FAKE_SERVER, decode_responses=True)
    import redis
    return redis.Redis.from_url(
        os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
    )


class JobQueue:
    def __init__(self, r=None):
        self.r = r or get_redis()

    def enqueue(self, job_type: str, payload: dict,
                idempotency_key: str | None = None) -> str | None:
        if idempotency_key:
            # NX: only the first enqueue with this key wins.
            if not self.r.set(DONE_PREFIX + idempotency_key, "1", nx=True, ex=DONE_TTL):
                return None
        job_id = str(uuid.uuid4())
        self.r.lpush(QUEUE_KEY, json.dumps(
            {"id": job_id, "type": job_type, "payload": payload}))
        return job_id

    def dequeue(self, timeout: int = 5) -> dict | None:
        try:
            item = self.r.brpop(QUEUE_KEY, timeout=timeout)
        except (RedisTimeoutError, TimeoutError):
            # Against real Redis/Valkey a blocking read on an idle queue can
            # surface as a socket read timeout rather than a nil reply. That is
            # not an error — it just means "no job"; let the caller poll again.
            return None
        if item is None:
            return None
        return json.loads(item[1])

    def depth(self) -> int:
        return self.r.llen(QUEUE_KEY)

    # interaction counter → reflection trigger
    def bump_interaction(self, project_id: str, user_id: str, agent_id: str) -> int:
        return self.r.incr(f"memgram:count:{project_id}:{user_id}:{agent_id}")
