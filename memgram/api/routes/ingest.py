"""POST /v1/ingest — the async post-hook target. Logs the interaction to
episodic memory and records job intents in the transactional OUTBOX, all in one
Postgres commit; jobs are then pushed to the queue immediately (fast path) or by
the scheduler's relay sweep if the process dies in between. No turn can exist
without its jobs, and no job can exist without its turn.

Also triggers the summarizer when the conversation is large: when the rough
token estimate of the messages crosses `summarize_threshold × context_limit`,
a `summarize` job rides in the same outbox commit."""
import logging
import os

from fastapi import APIRouter, Request
from pydantic import BaseModel

from memgram.worker import outbox

logger = logging.getLogger("memgram.api")

router = APIRouter()

# ~4 chars/token is the standard rough estimate; good enough for a trigger.
_CONTEXT_LIMIT = int(os.environ.get("MEMGRAM_CONTEXT_LIMIT", "26000"))
_SUMMARIZE_THRESHOLD = float(os.environ.get("MEMGRAM_SUMMARIZE_THRESHOLD", "0.6"))


class IngestBody(BaseModel):
    project_id: str
    agent_id: str
    user_id: str
    messages: list[dict]
    response_text: str | None = None
    # Per-request override of the worker's contradiction setting (None = use the
    # worker default). Lets the eval A/B v2 integration on ONE running stack.
    contradiction: bool | None = None


def _est_tokens(messages: list[dict]) -> int:
    chars = sum(len(m["content"]) for m in messages
                if isinstance(m.get("content"), str))
    return chars // 4


@router.post("", status_code=202)
async def ingest(request: Request, body: IngestBody):
    # Act only on a COMPLETED turn (the model produced a final answer). An
    # intermediate tool round-trip has no response_text; we skip it entirely
    # because the completed turn's message list already carries the full exchange
    # (user + tool_call + tool_result). This means a K-call tool loop logs and
    # extracts ONCE, not K times — bulk controlled at the source.
    if not body.response_text:
        return {"queued": False, "reason": "incomplete turn (tool round-trip)"}

    # Log only the CURRENT exchange — from the last user message to the end — not
    # the whole accumulated history an app may resend each turn.
    msgs = body.messages
    last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=0)
    entries = [(m["role"], m["content"]) for m in msgs[last_user:]
               if m.get("role") in ("user", "tool_call", "tool_result")
               and isinstance(m.get("content"), str)]
    entries.append(("assistant", body.response_text))

    jobs = [("extract", body.model_dump())]
    # Auto-trigger the summarizer when the conversation is over budget.
    if _est_tokens(body.messages) > _SUMMARIZE_THRESHOLD * _CONTEXT_LIMIT:
        jobs.append(("summarize", body.model_dump()))

    # ONE transaction: turn + job intents (the outbox pattern).
    ids = await request.app.state.store.ingest_turn(
        body.project_id, body.agent_id, body.user_id, entries, jobs)

    # Fast path: push to the queue now. If it fails (queue down) the turn is
    # already safe in Postgres — the scheduler's relay sweep dispatches it, so
    # we still 202. Memory ingestion must never surface infra blips to the app.
    try:
        await outbox.dispatch(request.app.state.pool, request.app.state.queue, ids=ids)
    except Exception:
        logger.warning("ingest fast-path dispatch failed; relay will recover", exc_info=True)

    return {"queued": True, "outbox_ids": ids}
