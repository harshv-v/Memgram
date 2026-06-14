"""Summarizer — compresses a long run of raw turns into a structured session
summary, written back to episodic memory as a single `compressed_session` row.
Token savings on long conversations: ~26k -> ~4k (~85%).

This is the only background agent the design treats as potentially *synchronous*
(it must finish before an over-threshold prompt is sent). Here it runs as a
normal queued job; the SDK can also call it inline when context is over budget.
The output is deterministic structured JSON, so callers parse, never regex.
"""
from memgram.agents.base import BaseAgent, fast_model

_SYSTEM = """You compress a long conversation between a user and an AI assistant into a compact, lossless-of-intent summary.
Return ONLY a JSON object with this exact shape:
{
  "key_decisions":       ["..."],
  "facts_established":    ["..."],
  "errors_corrected":     ["..."],
  "preferences_revealed": ["..."],
  "open_threads":         ["..."]
}
Rules:
- Capture everything a future session would need to continue seamlessly.
- Each item is one short, self-contained sentence.
- Drop pleasantries and redundancy. Empty arrays are fine."""

_SECTIONS = [
    ("key_decisions", "Key decisions"),
    ("facts_established", "Facts established"),
    ("errors_corrected", "Errors corrected"),
    ("preferences_revealed", "Preferences revealed"),
    ("open_threads", "Open threads"),
]


class SummarizerAgent(BaseAgent):
    def __init__(self, store, config=None, llm=None):
        super().__init__(store, config, llm)
        if not (config or {}).get("model"):
            self.model = fast_model()

    def build_prompt(self, job: dict) -> list[dict]:
        convo = "\n".join(
            f"{m['role']}: {m['content']}" for m in job["messages"]
            if isinstance(m.get("content"), str)
        )
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Conversation to compress:\n{convo}"},
        ]

    def parse_output(self, raw: str) -> dict:
        data = self.parse_json(raw)
        return {k: [s for s in data.get(k, []) if isinstance(s, str) and s.strip()]
                for k, _ in _SECTIONS}

    @staticmethod
    def render(summary: dict) -> str:
        """Flatten the structured summary into the text stored in episodic memory
        and (optionally) re-injected in place of the raw turns it replaced."""
        parts = []
        for key, label in _SECTIONS:
            items = summary.get(key) or []
            if items:
                parts.append(f"{label}:\n" + "\n".join(f"  - {i}" for i in items))
        return "## Session summary\n" + "\n".join(parts) if parts else ""

    async def on_success(self, job: dict, result: dict) -> None:
        text = self.render(result)
        if not text:
            return
        await self.store.log_episodic(
            project_id=job["project_id"], agent_id=job["agent_id"],
            user_id=job["user_id"], role="compressed_session", content=text,
        )
