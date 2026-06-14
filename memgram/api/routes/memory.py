"""Memory retrieval + management + GDPR endpoints."""
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/search")
async def search(request: Request, project_id: str, agent_id: str, user_id: str,
                 query: str, limit: int = 5):
    memories = await request.app.state.retriever.search(
        project_id=project_id, agent_id=agent_id, user_id=user_id,
        query=query, limit=min(limit, 20),
    )
    return {"memories": memories}


@router.get("")
async def list_memories(request: Request, project_id: str, agent_id: str,
                        user_id: str, tier: str | None = None, limit: int = 100):
    q = """
        SELECT id, content, memory_type, source, memory_tier, retention_score,
               reinforcement_count, emotional_weight, stability,
               last_accessed_at, created_at
        FROM semantic_memories
        WHERE project_id = $1 AND agent_id = $2 AND user_id = $3
    """
    args: list = [project_id, agent_id, user_id]
    if tier:
        q += " AND memory_tier = $4"
        args.append(tier)
    q += f" ORDER BY last_accessed_at DESC LIMIT {min(int(limit), 500)}"
    async with request.app.state.pool.acquire() as conn:
        rows = await conn.fetch(q, *args)
    return {"memories": [{**dict(r), "id": str(r["id"])} for r in rows]}


@router.delete("/{memory_id}", status_code=204)
async def delete_memory(request: Request, memory_id: UUID):
    async with request.app.state.pool.acquire() as conn:
        deleted = await conn.fetchval(
            "DELETE FROM semantic_memories WHERE id = $1 RETURNING id", memory_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail="Memory not found")


# -- GDPR ---------------------------------------------------------------------
@router.get("/export/{user_id}")
async def export_user(request: Request, user_id: str, project_id: str):
    """GET /v1/memory/export/:user_id — all memory as JSON."""
    out = {}
    async with request.app.state.pool.acquire() as conn:
        for table in ("instructions", "semantic_memories", "episodic_logs"):
            rows = await conn.fetch(
                f"SELECT * FROM {table} WHERE user_id = $1 AND project_id = $2",
                user_id, project_id)
            out[table] = [
                {k: (str(v) if k == "id" or k == "embedding" else v)
                 for k, v in dict(r).items()} for r in rows
            ]
    return out


@router.delete("/user/{user_id}", status_code=204)
async def forget_user(request: Request, user_id: str, project_id: str):
    """Hard delete across all tables — the right to be forgotten."""
    async with request.app.state.pool.acquire() as conn:
        for table in ("instructions", "semantic_memories", "episodic_logs"):
            await conn.execute(
                f"DELETE FROM {table} WHERE user_id = $1 AND project_id = $2",
                user_id, project_id)
