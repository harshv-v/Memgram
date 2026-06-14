"""Layered config (section 9). Identity + connection live as typed fields; the
behavioural config (features, decay, reflection, budgets, scope, webhooks) is
resolved from a preset + overrides into `settings` via memgram.presets.resolve.
"""
from typing import Any

from pydantic import BaseModel, Field

from memgram.presets import resolve


class MemgramConfig(BaseModel):
    # identity + connection
    api_key: str
    agent_name: str
    project_id: str | None = None          # defaults to agent_name if not given
    api_base_url: str = "http://localhost:8000"
    timeout: float = Field(default=5.0, gt=0)

    # behavioural config
    preset: str | None = None              # minimal|chatbot|coding|enterprise|privacy|custom
    overrides: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)  # resolved; do not set directly

    def model_post_init(self, __context) -> None:
        if self.project_id is None:
            self.project_id = self.agent_name
        # defaults -> preset -> developer overrides, merged once at init
        self.settings = resolve(self.preset, self.overrides)

    # convenience accessors used on the hot path
    @property
    def features(self) -> dict:
        return self.settings.get("features", {})

    @property
    def memory_budget(self) -> int:
        return int(self.settings.get("memory_budget", 4000))

    @property
    def context_limit(self) -> int:
        return int(self.settings.get("context_limit", 26000))

    @property
    def summarize_threshold(self) -> float:
        return float(self.settings.get("summarize_threshold", 0.6))

    @property
    def sharing_scope(self) -> str:
        return self.settings.get("sharing_scope", "project")
