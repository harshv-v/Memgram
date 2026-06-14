"""Instruction store CRUD — the Week 1 core.

Trust model (enforced here, at the API level):
  - source='user' rows may be created with status='active'.
  - source='agent_proposed' rows are ALWAYS forced to status='pending'.
    There is no code path that lets an agent write an active instruction.
"""
import json
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()

# Instruction cache — the hot path reads instructions on every LLM call, and
# they change rarely. 2-minute TTL in Redis/Valkey; flushed on any write.
_CACHE_TTL = 120
_CACHE_PREFIX = "memgram:instr:"


def _cache_key(project_id, agent_id, user_id, status):
    return f"{_CACHE_PREFIX}{project_id}:{agent_id}:{user_id}:{status}"


def _redis(request: Request):
    try:
        return request.app.state.queue.r
    except Exception:
        return None


def _cache_invalidate(request: Request):
    r = _redis(request)
    if r is None:
        return
    try:
        for k in r.scan_iter(match=_CACHE_PREFIX + "*"):
            r.delete(k)
    except Exception:
        pass  # cache is best-effort; never break a write on it


class InstructionCreate(BaseModel):
    project_id: str
    agent_id: str
    user_id: str
    content: str
    priority: int = Field(default=2, ge=1, le=3)
    source: str = Field(default="user", pattern="^(user|agent_proposed)$")


class InstructionPatch(BaseModel):
    content: str | None = None
    priority: int | None = Field(default=None, ge=1, le=3)
    status: str | None = Field(default=None, pattern="^(active|pending|rejected)$")


def _row(r) -> dict:
    d = dict(r)
    d["id"] = str(d["id"])
    return d


@router.get("")
async def list_instructions(
    request: Request,
    project_id: str,
    agent_id: str,
    user_id: str,
    status: str = "active",
):
    key = _cache_key(project_id, agent_id, user_id, status)
    r = _redis(request)
    if r is not None:
        try:
            hit = r.get(key)
            if hit is not None:
                return {"instructions": json.loads(hit), "cached": True}
        except Exception:
            pass
    async with request.app.state.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, project_id, agent_id, user_id, content, priority,
                   source, status, confidence, last_confirmed_at, created_at
            FROM instructions
            WHERE project_id = $1 AND agent_id = $2 AND user_id = $3 AND status = $4
            ORDER BY priority ASC, created_at ASC
            """,
            project_id, agent_id, user_id, status,
        )
    data = [_row(r) for r in rows]
    if r is not None:
        try:
            r.setex(key, _CACHE_TTL, json.dumps(data, default=str))
        except Exception:
            pass
    return {"instructions": data}


@router.post("", status_code=201)
async def create_instruction(request: Request, body: InstructionCreate):
    # Trust gate: agents can only propose; never write active instructions.
    status = "active" if body.source == "user" else "pending"
    async with request.app.state.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO instructions (project_id, agent_id, user_id, content, priority, source, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id, project_id, agent_id, user_id, content, priority,
                      source, status, confidence, last_confirmed_at, created_at
            """,
            body.project_id, body.agent_id, body.user_id,
            body.content, body.priority, body.source, status,
        )
    _cache_invalidate(request)
    return _row(row)


@router.patch("/{instruction_id}")
async def patch_instruction(request: Request, instruction_id: UUID, body: InstructionPatch):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    sets = ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(updates))
    async with request.app.state.pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            UPDATE instructions SET {sets}, last_confirmed_at = NOW()
            WHERE id = $1
            RETURNING id, project_id, agent_id, user_id, content, priority,
                      source, status, confidence, last_confirmed_at, created_at
            """,
            instruction_id, *updates.values(),
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Instruction not found")
    _cache_invalidate(request)
    return _row(row)


@router.delete("/{instruction_id}", status_code=204)
async def delete_instruction(request: Request, instruction_id: UUID):
    async with request.app.state.pool.acquire() as conn:
        deleted = await conn.fetchval(
            "DELETE FROM instructions WHERE id = $1 RETURNING id", instruction_id
        )
    if deleted is None:
        raise HTTPException(status_code=404, detail="Instruction not found")
    _cache_invalidate(request)
