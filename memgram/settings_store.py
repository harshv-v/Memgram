"""Runtime project settings — toggles you can flip WITHOUT a restart.

Resolution order: project_settings row (DB) > environment default. Values are
cached in-process for a short TTL so the hot path never adds a query per call.
Setting keys are whitelisted; no free-form config through the API.
"""
import os
import time

SETTING_KEYS = {
    "pii_redact": {"type": "bool", "env": "MEMGRAM_PII_REDACT", "default": "0"},
    "sanitize":   {"type": "bool", "env": "MEMGRAM_SANITIZE", "default": "1"},
}
_TTL = float(os.environ.get("MEMGRAM_SETTINGS_TTL", "30"))
_cache: dict[tuple, tuple[float, str]] = {}


async def get_setting(pool, project_id: str, key: str) -> str:
    spec = SETTING_KEYS[key]
    ck = (project_id, key)
    hit = _cache.get(ck)
    if hit and time.monotonic() - hit[0] < _TTL:
        return hit[1]
    value = None
    try:
        async with pool.acquire() as conn:
            value = await conn.fetchval(
                "SELECT value FROM project_settings WHERE project_id=$1 AND key=$2",
                project_id, key)
    except Exception:
        pass  # settings lookup must never break a write path
    if value is None:
        value = os.environ.get(spec["env"], spec["default"])
    _cache[ck] = (time.monotonic(), value)
    return value


async def get_bool(pool, project_id: str, key: str) -> bool:
    return (await get_setting(pool, project_id, key)).lower() in ("1", "true", "on", "yes")


async def set_setting(pool, project_id: str, key: str, value: str) -> None:
    if key not in SETTING_KEYS:
        raise KeyError(key)
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO project_settings (project_id, key, value)
               VALUES ($1,$2,$3)
               ON CONFLICT (project_id, key)
               DO UPDATE SET value = $3, updated_at = NOW()""",
            project_id, key, value)
    _cache.pop((project_id, key), None)
