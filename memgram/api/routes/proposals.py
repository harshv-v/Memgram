"""Proposal review — the user side of the trust gate. Approve promotes a
pending agent-proposed instruction to active; reject keeps it forever
non-injectable. These are USER actions (dashboard / app UI)."""
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("")
async def list_proposals(request: Request, project_id: str, agent_id: str, user_id: str):
    async with request.app.state.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, priority, created_at FROM instructions
            WHERE project_id = $1 AND agent_id = $2 AND user_id = $3
              AND status = 'pending' AND source = 'agent_proposed'
            ORDER BY created_at DESC
            """,
            project_id, agent_id, user_id)
    return {"proposals": [{**dict(r), "id": str(r["id"])} for r in rows]}


async def _set_status(request: Request, proposal_id: UUID, status: str) -> dict:
    async with request.app.state.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE instructions SET status = $2, last_confirmed_at = NOW()
            WHERE id = $1 AND source = 'agent_proposed' AND status IN ('pending', 'review')
            RETURNING id, content, status
            """,
            proposal_id, status)
    if row is None:
        raise HTTPException(status_code=404, detail="No reviewable proposal with that id")
    return {**dict(row), "id": str(row["id"])}


@router.post("/{proposal_id}/approve")
async def approve(request: Request, proposal_id: UUID):
    return await _set_status(request, proposal_id, "active")


@router.post("/{proposal_id}/reject")
async def reject(request: Request, proposal_id: UUID):
    return await _set_status(request, proposal_id, "rejected")
