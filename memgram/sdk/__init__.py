"""Memgram SDK — wrap your LLM call, memory works instantly."""
from typing import Any

from memgram.sdk.assembler import ContextAssembler
from memgram.sdk.client import MemgramAPIClient
from memgram.sdk.config import MemgramConfig
from memgram.sdk.proxy import WrappedClient


class Memgram:
    def __init__(self, api_key: str, agent_name: str, project_id: str | None = None,
                 api_base_url: str = "http://localhost:8000",
                 preset: str | None = None, **overrides: Any):
        """`preset` selects a bundle (minimal|chatbot|coding|enterprise|privacy|
        custom); any extra keyword (features=..., decay=..., memory_budget=...,
        sharing_scope=..., webhooks=...) overrides it. See memgram.presets."""
        self.config = MemgramConfig(
            api_key=api_key, agent_name=agent_name,
            project_id=project_id, api_base_url=api_base_url,
            preset=preset, overrides=overrides,
        )
        self.api = MemgramAPIClient(self.config)
        self._assembler = ContextAssembler(self.api, self.config)

    def wrap(self, client, agent_name: str | None = None):
        """Wrap any LLM client. One line. Nothing else changes.
        OpenAI-compatible clients (OpenAI, Ollama, Groq, Gemini's OpenAI
        endpoint, vLLM, ...) are detected by `.chat`; native Anthropic clients
        by `.messages`."""
        config = self.config
        if agent_name is not None:
            config = config.model_copy(update={"agent_name": agent_name})
        if hasattr(client, "chat"):
            return WrappedClient(client, ContextAssembler(self.api, config), config)
        if hasattr(client, "messages"):
            from memgram.sdk.anthropic_proxy import AnthropicWrappedClient
            return AnthropicWrappedClient(client, self.api, config)
        raise TypeError(
            "Don't know how to wrap this client: expected an OpenAI-compatible "
            "client (has .chat) or an Anthropic client (has .messages).")

    @property
    def settings(self) -> dict:
        return self.config.settings

    # Convenience for apps (and the demo) to store a user instruction.
    def add_instruction(self, user_id: str, content: str, priority: int = 2) -> dict:
        return self.api.create_instruction(
            user_id=user_id, agent_id=self.config.agent_name,
            content=content, priority=priority, source="user",
        )
