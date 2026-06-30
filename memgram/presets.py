"""Configuration presets and the three-layer merge (section 9 of the design doc).

Layer order (lower overrides higher):
  1. platform defaults  (DEFAULTS below)
  2. a named preset      (minimal | chatbot | coding | enterprise | privacy | custom)
  3. developer overrides (whatever they pass to Memgram)

`resolve(preset, overrides)` returns one normalized, deep-merged dict that both
the SDK and the worker read — so a feature flag set once is honored everywhere.
Shared by `sdk/config.py` (hot path) and `worker/__main__.py` (background).
"""
from __future__ import annotations

import copy

# ── platform defaults — sensible for ~80% of cases ───────────────────────────
DEFAULTS: dict = {
    "context_limit": 26000,
    "memory_budget": 4000,
    "summarize_threshold": 0.6,   # compress when context is this fraction full
    "raw_turns_keep": 10,
    "sharing_scope": "private",   # global | project | private (isolated by default)
    "features": {
        "semantic": True, "episodic": True, "procedural": False,
        "instructions": True, "reflection": True, "decay": True,
        "summarizer": True, "proposals": True,
    },
    "decay": {
        "episodic_stability": 0.5, "semantic_stability": 2.0,
        "procedural_stability": 3.0, "archive_threshold": 0.1,
        "habit_threshold": 7, "demotion_days": 90,
    },
    "reflection": {"trigger_every_n": 20, "trigger_every_hrs": 24},
    "user_can": {
        "view_memories": True, "edit_instructions": True, "delete_memories": True,
        "approve_proposals": True, "export_data": True, "delete_all": True,
    },
    "webhooks": {},
}

# ── presets — only the deltas from DEFAULTS ──────────────────────────────────
PRESETS: dict = {
    "minimal": {  # instruction store only, zero background cost
        "features": {"semantic": False, "episodic": False, "procedural": False,
                     "reflection": False, "decay": False, "summarizer": False,
                     "proposals": False},
    },
    "chatbot": {  # semantic + episodic + instructions, daily reflection
        "features": {"procedural": False},
        "reflection": {"trigger_every_hrs": 24},
    },
    "coding": {  # all four pillars, procedural on, slower-to-habit
        "features": {"procedural": True},
        "decay": {"habit_threshold": 10},
    },
    "enterprise": {  # full stack, locked-down user permissions, admin approval
        "user_can": {"delete_memories": False, "delete_all": False},
        "audit_log": True, "admin_only_proposals": True,
    },
    "privacy": {  # self-hosted, hard deletes, nothing leaves your infra
        "self_hosted": True, "hard_deletes": True,
    },
    "custom": {},  # everything exposed, no opinions
}

VALID_PRESETS = set(PRESETS)
_VALID_SCOPES = {"global", "project", "private"}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve(preset: str | None = None, overrides: dict | None = None) -> dict:
    """Defaults → preset → overrides, deep-merged. Raises on an unknown preset
    or scope so misconfiguration fails loudly at init, not silently at runtime."""
    cfg = copy.deepcopy(DEFAULTS)
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(VALID_PRESETS)}")
        cfg = _deep_merge(cfg, PRESETS[preset])
    if overrides:
        cfg = _deep_merge(cfg, {k: v for k, v in overrides.items() if v is not None})
    if cfg["sharing_scope"] not in _VALID_SCOPES:
        raise ValueError(f"sharing_scope must be one of {sorted(_VALID_SCOPES)}")
    return cfg
