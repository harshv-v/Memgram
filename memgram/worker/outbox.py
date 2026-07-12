"""Transactional-outbox relay.

The API writes (episodic rows + outbox rows) in one Postgres transaction, then
calls `dispatch` immediately — the happy path adds ~one UPDATE. If the process
dies between commit and enqueue, the row is still there with
dispatched_at IS NULL, and the scheduler's sweep re-dispatches it.

Exactly-once effect: every enqueue uses idempotency_key=f"outbox:{row.id}", so
a row that was enqueued but not yet marked dispatched can be re-dispatched
safely — the queue's SET NX drops the duplicate.
"""
import json
import logging

logger = logging.getLogger("memgram.worker")


async def dispatch(pool, queue, ids: list | None = None,
                   older_than_s: float = 0, limit: int = 200) -> int:
    """Enqueue pending outbox rows and mark them dispatched. Returns count.
    With `ids`, dispatches exactly those rows (API fast path); otherwise sweeps
    everything pending older than `older_than_s` (relay path)."""
    async with pool.acquire() as conn:
        if ids:
            rows = await conn.fetch(
                "SELECT id, job_type, payload FROM outbox "
                "WHERE id = ANY($1::uuid[]) AND dispatched_at IS NULL", ids)
        else:
            rows = await conn.fetch(
                "SELECT id, job_type, payload FROM outbox "
                "WHERE dispatched_at IS NULL AND created_at < NOW() - ($1 || ' seconds')::interval "
                "ORDER BY created_at LIMIT $2", str(older_than_s), limit)
        n = 0
        for r in rows:
            payload = r["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            queue.enqueue(r["job_type"], payload, idempotency_key=f"outbox:{r['id']}")
            await conn.execute(
                "UPDATE outbox SET dispatched_at = NOW() WHERE id = $1", r["id"])
            n += 1
        if n and not ids:
            logger.info("outbox relay: re-dispatched %d stranded job(s)", n)
    return n
