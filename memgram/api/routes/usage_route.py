"""GET /v1/usage — everything the memory layer consumed, per user.

Answers, in one call: how many tokens has memory cost this user (by source:
extraction, verification, reflection, proposals, summarization, injection),
what's the estimated dollar cost, and how much STORAGE their memory occupies
(rows + bytes by tier). The dashboard's usage panel reads this.
"""
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("")
async def get_usage(request: Request, project_id: str, user_id: str,
                    agent_id: str | None = None, days: int = 30):
    pool = request.app.state.pool
    days = max(1, min(days, 365))
    extra = " AND agent_id = $3" if agent_id else ""
    args = [project_id, user_id] + ([agent_id] if agent_id else [])

    async with pool.acquire() as conn:
        tok = await conn.fetch(
            f"""SELECT kind, model, SUM(tokens_in)::bigint AS tin,
                       SUM(tokens_out)::bigint AS tout, SUM(cost_usd) AS cost,
                       COUNT(*) AS events
                FROM usage_events
                WHERE project_id=$1 AND user_id=$2{extra}
                  AND created_at > NOW() - INTERVAL '{days} days'
                GROUP BY kind, model ORDER BY cost DESC NULLS LAST""", *args)
        sem = await conn.fetch(
            f"""SELECT memory_tier, COUNT(*) AS rows,
                       COALESCE(SUM(pg_column_size(semantic_memories.*)), 0)::bigint AS bytes
                FROM semantic_memories
                WHERE project_id=$1 AND user_id=$2{extra} GROUP BY memory_tier""", *args)
        epi = await conn.fetchrow(
            f"""SELECT COUNT(*) AS rows,
                       COALESCE(SUM(pg_column_size(episodic_logs.*)), 0)::bigint AS bytes
                FROM episodic_logs WHERE project_id=$1 AND user_id=$2{extra}""", *args)
        ins = await conn.fetchval(
            f"SELECT COUNT(*) FROM instructions WHERE project_id=$1 AND user_id=$2{extra}",
            *args)

    by_kind = [{"kind": r["kind"], "model": r["model"], "tokens_in": r["tin"],
                "tokens_out": r["tout"], "cost_usd": float(r["cost"] or 0),
                "events": r["events"]} for r in tok]
    return {
        "window_days": days,
        "tokens": {
            "by_kind": by_kind,
            "total_in": sum(k["tokens_in"] for k in by_kind),
            "total_out": sum(k["tokens_out"] for k in by_kind),
            "injected": sum(k["tokens_in"] for k in by_kind if k["kind"] == "injection"),
            "total_cost_usd": round(sum(k["cost_usd"] for k in by_kind), 6),
        },
        "storage": {
            "semantic": {r["memory_tier"]: {"rows": r["rows"], "bytes": r["bytes"]}
                         for r in sem},
            "episodic": {"rows": epi["rows"], "bytes": epi["bytes"]},
            "instructions": ins,
            "total_bytes": sum(r["bytes"] for r in sem) + epi["bytes"],
        },
    }
