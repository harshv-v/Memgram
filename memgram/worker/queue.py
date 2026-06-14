"""Durable job queue on Redis/Valkey **Streams** with a consumer group.

Why Streams and not a plain list (LPUSH/BRPOP): a list pop is fire-and-forget —
if a worker dies after popping but before finishing, the job is lost. Streams give
us at-least-once delivery: a delivered-but-unacked entry stays in the group's
Pending Entries List (PEL) and is reclaimed by another worker (XAUTOCLAIM) once
its original consumer goes idle. We ack (and delete) only after a job succeeds;
a job that fails past MAX_DELIVERIES goes to a dead-letter stream for inspection.

Consumer groups also give horizontal scale for free: run N workers, each a
distinct consumer in the same group, and the stream load-balances across them.

Still ~one datastore, no Celery, no RabbitMQ. Set MEMGRAM_FAKE_REDIS=1 to run on
fakeredis (tests / no-infra dev) — it supports the full Streams surface we use.
"""
import json
import os
import socket
import uuid

try:
    from redis.exceptions import ResponseError, TimeoutError as RedisTimeoutError
except Exception:  # pragma: no cover - redis is always installed
    ResponseError = Exception
    RedisTimeoutError = ()

STREAM_KEY = "memgram:jobs"
DLQ_KEY = "memgram:jobs:dead"
GROUP = "memgram:workers"
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
    def __init__(self, r=None, consumer: str | None = None):
        self.r = r or get_redis()
        # unique, identifiable consumer name per process (host-pid-rand)
        self.consumer = consumer or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self._ensure_group()

    def _ensure_group(self) -> None:
        try:
            # MKSTREAM so the group can be created before the first job exists.
            self.r.xgroup_create(STREAM_KEY, GROUP, id="0", mkstream=True)
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):  # group already exists -> fine
                raise

    # -- produce -----------------------------------------------------------
    def enqueue(self, job_type: str, payload: dict,
                idempotency_key: str | None = None) -> str | None:
        if idempotency_key:
            # NX: only the first enqueue with this key wins (decay/reflect sweeps).
            if not self.r.set(DONE_PREFIX + idempotency_key, "1", nx=True, ex=DONE_TTL):
                return None
        job_id = str(uuid.uuid4())
        self.r.xadd(STREAM_KEY, {
            "id": job_id, "type": job_type, "payload": json.dumps(payload)})
        return job_id

    # -- consume -----------------------------------------------------------
    def dequeue(self, timeout: int = 5) -> dict | None:
        """Block for one new (never-delivered) job. The returned dict carries
        `msg_id`; the caller MUST ack() on success or it stays pending."""
        block_ms = max(1, int(timeout * 1000))
        try:
            resp = self.r.xreadgroup(
                GROUP, self.consumer, {STREAM_KEY: ">"}, count=1, block=block_ms)
        except (RedisTimeoutError, TimeoutError):
            return None
        if not resp:
            return None
        _stream, entries = resp[0]
        if not entries:
            return None
        msg_id, fields = entries[0]
        return self._job(msg_id, fields)

    def reclaim(self, min_idle_ms: int = 60_000, count: int = 16) -> list[dict]:
        """Steal entries whose consumer has been idle past `min_idle_ms` — i.e.
        a crashed/stuck worker's in-flight jobs. This is the durability payoff."""
        cursor, entries, _deleted = self.r.xautoclaim(
            STREAM_KEY, GROUP, self.consumer, min_idle_ms, "0", count=count)
        jobs = []
        for msg_id, fields in entries:
            if not fields:  # entry was deleted under us; clear it from the PEL
                self.r.xack(STREAM_KEY, GROUP, msg_id)
                continue
            jobs.append(self._job(msg_id, fields))
        return jobs

    def deliveries(self, job: dict) -> int:
        """How many times this entry has been delivered (for DLQ thresholding)."""
        try:
            pend = self.r.xpending_range(
                STREAM_KEY, GROUP, min=job["msg_id"], max=job["msg_id"], count=1)
            return int(pend[0]["times_delivered"]) if pend else 1
        except Exception:
            return 1

    def ack(self, job: dict) -> None:
        """Acknowledge AND delete — the job is done; drop it from the stream."""
        self.r.xack(STREAM_KEY, GROUP, job["msg_id"])
        self.r.xdel(STREAM_KEY, job["msg_id"])

    def dead_letter(self, job: dict, reason: str) -> None:
        self.r.xadd(DLQ_KEY, {
            "id": job.get("id", ""), "type": job.get("type", ""),
            "payload": json.dumps(job.get("payload", {})), "reason": reason})
        self.ack(job)  # remove from the live stream so it isn't redelivered

    # -- introspection -----------------------------------------------------
    def depth(self) -> int:
        """Outstanding entries (waiting + delivered-but-unacked). Acked jobs are
        deleted, so XLEN tracks real backlog."""
        return self.r.xlen(STREAM_KEY)

    def dlq_depth(self) -> int:
        try:
            return self.r.xlen(DLQ_KEY)
        except Exception:
            return 0

    # interaction counter → reflection trigger
    def bump_interaction(self, project_id: str, user_id: str, agent_id: str) -> int:
        return self.r.incr(f"memgram:count:{project_id}:{user_id}:{agent_id}")

    @staticmethod
    def _job(msg_id: str, fields: dict) -> dict:
        return {
            "msg_id": msg_id,
            "id": fields.get("id"),
            "type": fields["type"],
            "payload": json.loads(fields["payload"]),
        }
