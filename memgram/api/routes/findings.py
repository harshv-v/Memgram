"""GET /v1/findings — what the monitor agents found in stored memory.
POST /v1/findings/{id}/resolve — acknowledge a finding (human action)."""
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("")
async def list_findings(request: Request, project_id: str,
                        resolved: bool = False, limit: int = 200):
    async with request.app.state.pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, project_id, agent_id, user_id, monitor, kind, severity,
                      detail, memory_id, resolved, created_at
               FROM memory_findings WHERE project_id = $1 AND resolved = $2
               ORDER BY severity = 'critical' DESC, created_at DESC LIMIT $3""",
            project_id, resolved, min(limit, 1000))
    return {"findings": [
        {**dict(r), "id": str(r["id"]),
         "memory_id": str(r["memory_id"]) if r["memory_id"] else None}
        for r in rows]}


@router.post("/{finding_id}/resolve")
async def resolve(request: Request, finding_id: UUID):
    async with request.app.state.pool.acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE memory_findings SET resolved = TRUE WHERE id = $1 RETURNING id",
            finding_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Finding not found")
    return {"resolved": str(row["id"])}
