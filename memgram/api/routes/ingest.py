"""POST /v1/ingest — the async post-hook target. Logs the interaction to
episodic memory and enqueues an extract job. Returns 202 immediately; all
intelligence happens in the worker. This endpoint must stay dumb and fast.

Also triggers the summarizer when the conversation is large: when the rough
token estimate of the messages crosses `summarize_threshold × context_limit`,
a `summarize` job is enqueued (async). The compressed session is then available
for the next turn. The design treats the summarizer as potentially synchronous;
v1 runs it async to keep the post-hook non-blocking (a documented tradeoff)."""
import os

from fastapi import APIRouter, Request
from pydantic import BaseModel

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


def _est_tokens(messages: list[dict]) -> int:
    chars = sum(len(m["content"]) for m in messages
                if isinstance(m.get("content"), str))
    return chars // 4


@router.post("", status_code=202)
async def ingest(request: Request, body: IngestBody):
    store = request.app.state.store
    for m in body.messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            await store.log_episodic(body.project_id, body.agent_id, body.user_id,
                                     "user", m["content"])
    if body.response_text:
        await store.log_episodic(body.project_id, body.agent_id, body.user_id,
                                 "assistant", body.response_text)

    queue = request.app.state.queue
    job_id = queue.enqueue("extract", body.model_dump())

    # Auto-trigger the summarizer when the conversation is over budget.
    if _est_tokens(body.messages) > _SUMMARIZE_THRESHOLD * _CONTEXT_LIMIT:
        queue.enqueue("summarize", body.model_dump())

    return {"queued": True, "job_id": job_id}
