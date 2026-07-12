"""GET /v1/context — the hot path, in ONE round trip.

Previously the SDK made two sequential HTTP calls per LLM call (instructions,
then semantic search). This endpoint does both server-side IN PARALLEL and
returns one payload — halving hot-path network overhead. It also records the
injection: how many tokens Memgram is about to add to the developer's prompt,
attributed per (project, agent, user) for /v1/usage.
"""
import asyncio

from fastapi import APIRouter, Request

from memgram import usage

router = APIRouter()


@router.get("")
async def get_context(request: Request, project_id: str, agent_id: str,
                      user_id: str, query: str | None = None, limit: int = 5):
    pool = request.app.state.pool
    retriever = request.app.state.retriever

    async def instructions():
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT content, priority FROM instructions
                   WHERE project_id=$1 AND agent_id=$2 AND user_id=$3 AND status='active'
                   ORDER BY priority ASC, created_at ASC""",
                project_id, agent_id, user_id)
        return [dict(r) for r in rows]

    async def memories():
        if not query:
            return []
        return await retriever.search(project_id=project_id, agent_id=agent_id,
                                      user_id=user_id, query=query,
                                      limit=min(limit, 20))

    instr, mems = await asyncio.gather(instructions(), memories())

    injected = sum(usage.est_tokens(i["content"]) for i in instr) + \
        sum(usage.est_tokens(m["content"]) for m in mems)
    if injected:  # fire-and-forget: accounting never adds latency
        asyncio.get_running_loop().create_task(usage.record(
            pool, project_id, agent_id, user_id, "injection", tokens_in=injected))

    return {"instructions": instr, "memories": mems, "injected_tokens_est": injected}
