"""Monitor agents — a second pair of eyes on STORED memory.

The pipeline defends data on the way in (faithfulness gate, sanitizer, dedup);
the monitors audit what actually LANDED, continuously. Three of them, all
deterministic SQL/regex — zero LLM cost, no hardcoded behavior: every pattern
comes from memgram.safety (single source of truth) and every threshold is
config/env-driven.

  SafetyMonitor   injection-shaped content at rest; PII at rest
  HygieneMonitor  duplicate active content; per-user memory bloat
  DriftMonitor    active-but-long-unaccessed rows (decay lag); stuck reviews

Findings land in memory_findings (GET /v1/findings) and gauge
memgram_open_findings. Each run replaces its own unresolved findings, so the
table reflects the CURRENT state, not an ever-growing log.
"""
import os

from memgram.obs import Gauge
from memgram.safety import _INJECTION_RE, _PII_RES

OPEN_FINDINGS = Gauge("memgram_open_findings", "Unresolved monitor findings",
                      ["monitor", "kind"])


def _cfg(config: dict, key: str, env: str, default):
    v = (config or {}).get(key, os.environ.get(env, default))
    return type(default)(v)


class _Monitor:
    """Shared machinery: replace-my-own-findings semantics + gauge updates."""
    name = "monitor"

    def __init__(self, store, config: dict | None = None):
        self.store = store
        self.config = config or {}

    async def run(self, job: dict) -> dict:
        findings = await self.scan()
        async with self.store.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM memory_findings WHERE monitor = $1 AND NOT resolved",
                    self.name)
                for f in findings:
                    await conn.execute(
                        """INSERT INTO memory_findings
                           (project_id, agent_id, user_id, monitor, kind, severity,
                            detail, memory_id)
                           VALUES ($1,$2,$3,$4,$5,$6,$7,$8)""",
                        f["project_id"], f["agent_id"], f["user_id"], self.name,
                        f["kind"], f["severity"], f["detail"], f.get("memory_id"))
        kinds: dict = {}
        for f in findings:
            kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
        for kind, n in kinds.items():
            OPEN_FINDINGS.labels(monitor=self.name, kind=kind).set(n)
        return {"monitor": self.name, "findings": len(findings), "by_kind": kinds}

    async def scan(self) -> list[dict]:  # pragma: no cover - abstract
        raise NotImplementedError


class SafetyMonitor(_Monitor):
    """Content that should never sit in memory: instruction-shaped text
    (an injection that got past the write path, or predates the sanitizer)
    and PII at rest. Patterns come from memgram.safety — one source of truth."""
    name = "safety"

    async def scan(self) -> list[dict]:
        async with self.store.pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, project_id, agent_id, user_id, content
                   FROM semantic_memories WHERE memory_tier != 'archived'""")
        out = []
        for r in rows:
            if _INJECTION_RE.search(r["content"]):
                out.append({**_ids(r), "kind": "injection_suspect", "severity": "critical",
                            "memory_id": r["id"],
                            "detail": f"instruction-shaped memory content: {r['content'][:120]!r}"})
            for rx, label in _PII_RES:
                if rx.search(r["content"]):
                    out.append({**_ids(r), "kind": "pii_at_rest", "severity": "warn",
                                "memory_id": r["id"],
                                "detail": f"{label} detected in stored memory {r['id']}"})
                    break
        return out


class HygieneMonitor(_Monitor):
    """Redundancy and bloat: identical active content that escaped dedup, and
    users whose active memory count exceeds the bloat threshold."""
    name = "hygiene"

    async def scan(self) -> list[dict]:
        bloat_at = _cfg(self.config, "bloat_threshold", "MEMGRAM_MONITOR_BLOAT", 500)
        async with self.store.pool.acquire() as conn:
            dups = await conn.fetch(
                """SELECT project_id, agent_id, user_id, content, COUNT(*) AS n
                   FROM semantic_memories WHERE memory_tier != 'archived'
                   GROUP BY project_id, agent_id, user_id, content HAVING COUNT(*) > 1""")
            bloat = await conn.fetch(
                """SELECT project_id, agent_id, user_id, COUNT(*) AS n
                   FROM semantic_memories WHERE memory_tier != 'archived'
                   GROUP BY project_id, agent_id, user_id HAVING COUNT(*) > $1""",
                bloat_at)
        out = [{**_ids(r), "kind": "duplicate_content", "severity": "warn",
                "detail": f"{r['n']} identical active memories: {r['content'][:100]!r}"}
               for r in dups]
        out += [{**_ids(r), "kind": "memory_bloat", "severity": "info",
                 "detail": f"{r['n']} active memories (threshold {bloat_at}) — "
                           "check extraction volume / decay settings"}
                for r in bloat]
        return out


class DriftMonitor(_Monitor):
    """Lifecycle lag: rows still 'active' despite long inactivity (decay job
    not running?) and agent-proposed instructions stuck in review."""
    name = "drift"

    async def scan(self) -> list[dict]:
        stale_days = _cfg(self.config, "stale_days", "MEMGRAM_MONITOR_STALE_DAYS", 14)
        review_days = _cfg(self.config, "review_days", "MEMGRAM_MONITOR_REVIEW_DAYS", 14)
        async with self.store.pool.acquire() as conn:
            stale = await conn.fetch(
                """SELECT id, project_id, agent_id, user_id FROM semantic_memories
                   WHERE memory_tier = 'active'
                     AND last_accessed_at < NOW() - ($1 || ' days')::interval""",
                str(stale_days))
            stuck = await conn.fetch(
                """SELECT id, project_id, agent_id, user_id FROM instructions
                   WHERE status IN ('pending', 'review')
                     AND created_at < NOW() - ($1 || ' days')::interval""",
                str(review_days))
        out = [{**_ids(r), "kind": "stale_active", "severity": "info", "memory_id": r["id"],
                "detail": f"active but unaccessed for {stale_days}+ days — is the decay job running?"}
               for r in stale]
        out += [{**_ids(r), "kind": "review_stuck", "severity": "warn",
                 "detail": f"instruction awaiting human review for {review_days}+ days"}
                for r in stuck]
        return out


def _ids(row) -> dict:
    return {"project_id": row["project_id"], "agent_id": row["agent_id"],
            "user_id": row["user_id"]}


MONITORS = (SafetyMonitor, HygieneMonitor, DriftMonitor)


class MonitorSuite:
    """The 'monitor' job: run every monitor, return the combined report."""

    def __init__(self, store, config: dict | None = None):
        self.monitors = [m(store, config) for m in MONITORS]

    async def run(self, job: dict) -> dict:
        results = [await m.run(job) for m in self.monitors]
        return {"monitors": results,
                "total_findings": sum(r["findings"] for r in results)}
