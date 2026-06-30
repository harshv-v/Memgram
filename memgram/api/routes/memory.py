"""Memory retrieval + management + GDPR/portability endpoints."""
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from memgram.memory.embedder import to_pgvector

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
               reinforcement_count, emotional_weight, stability, scope,
               (superseded_by IS NOT NULL) AS superseded,
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


# -- Portability ("memory on a pendrive") -------------------------------------
class ImportBody(BaseModel):
    data: dict                          # an /export bundle
    target_project_id: str | None = None  # optional remap on load
    target_user_id: str | None = None


@router.post("/import")
async def import_memory(request: Request, body: ImportBody):
    """Load an exported memory bundle into THIS instance. Semantic memories are
    RE-EMBEDDED with this instance's embedder, so a bundle exported at one vector
    dimension (e.g. 384-d local) imports cleanly into another (e.g. 1536-d OpenAI)
    — the content is portable, the vectors are regenerated. The memory's learned
    strength (reinforcement_count, stability, tier) is preserved."""
    data, embedder = body.data, request.app.state.embedder
    counts = {"instructions": 0, "semantic_memories": 0, "episodic_logs": 0}

    async with request.app.state.pool.acquire() as conn:
        for r in data.get("instructions", []):
            await conn.execute(
                """INSERT INTO instructions
                   (project_id, agent_id, user_id, content, priority, source, status)
                   VALUES ($1,$2,$3,$4,$5,$6,$7)""",
                body.target_project_id or r["project_id"], r["agent_id"],
                body.target_user_id or r["user_id"], r["content"],
                r.get("priority", 2), r.get("source", "user"), r.get("status", "active"))
            counts["instructions"] += 1

        for r in data.get("semantic_memories", []):
            emb = to_pgvector(await embedder.embed(r["content"]))  # re-embed = dim-safe
            await conn.execute(
                """INSERT INTO semantic_memories
                   (project_id, agent_id, user_id, content, memory_type, source,
                    embedding, stability, reinforcement_count, retention_score,
                    emotional_weight, memory_tier, scope)
                   VALUES ($1,$2,$3,$4,$5,$6,$7::vector,$8,$9,$10,$11,$12,$13)""",
                body.target_project_id or r["project_id"], r["agent_id"],
                body.target_user_id or r["user_id"], r["content"],
                r.get("memory_type"), r.get("source", "import"), emb,
                r.get("stability", 2.0), r.get("reinforcement_count", 1),
                r.get("retention_score", 1.0), r.get("emotional_weight", 1.0),
                r.get("memory_tier", "active"), r.get("scope", "project"))
            counts["semantic_memories"] += 1

        for r in data.get("episodic_logs", []):
            await conn.execute(
                """INSERT INTO episodic_logs (project_id, agent_id, user_id, role, content)
                   VALUES ($1,$2,$3,$4,$5)""",
                body.target_project_id or r["project_id"], r["agent_id"],
                body.target_user_id or r["user_id"], r.get("role", "user"), r["content"])
            counts["episodic_logs"] += 1

    return {"imported": counts}
