"""Extractor — runs async after every interaction. Pulls facts, preferences,
entities, and corrections out of the conversation and writes them to semantic
memory (with dedup → reinforcement). Corrections get emotional_weight 2.0 so
mistakes persist longer than neutral memories."""
from memgram.agents.base import BaseAgent, fast_model

_SYSTEM = """You extract long-term memories from a conversation between a user and an AI assistant.
Return ONLY a JSON object with this exact shape:
{
  "facts":        [{"content": "..."}],
  "preferences":  [{"content": "..."}],
  "entities":     [{"content": "..."}],
  "corrections":  [{"content": "..."}]
}
Rules:
- Only include things worth remembering ACROSS sessions (identity, projects, stack, stable preferences, corrections of the assistant's mistakes).
- Skip pleasantries, one-off context, and anything ephemeral.
- Each item must be a single self-contained sentence, stated about "the user" in third person.
- corrections = moments where the user corrected the assistant or expressed frustration at a mistake.
- Empty arrays are fine. Quality over quantity."""


class ExtractorAgent(BaseAgent):
    def __init__(self, store, config=None, llm=None):
        super().__init__(store, config, llm)
        if not (config or {}).get("model"):
            self.model = fast_model()

    def build_prompt(self, job: dict) -> list[dict]:
        convo = "\n".join(
            f"{m['role']}: {m['content']}" for m in job["messages"]
            if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str)
        )
        if job.get("response_text"):
            convo += f"\nassistant: {job['response_text']}"
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Conversation:\n{convo}"},
        ]

    def parse_output(self, raw: str) -> dict:
        data = self.parse_json(raw)
        out = {}
        for key, mtype in [("facts", "fact"), ("preferences", "preference"),
                           ("entities", "entity"), ("corrections", "correction")]:
            out[key] = [
                {"content": i["content"], "memory_type": mtype}
                for i in data.get(key, []) if i.get("content")
            ]
        return out

    async def on_success(self, job: dict, result: dict) -> None:
        ids = job  # project/agent/user ids ride on the job
        for key, items in result.items():
            for item in items:
                await self.store.upsert_semantic(
                    project_id=ids["project_id"], agent_id=ids["agent_id"],
                    user_id=ids["user_id"], content=item["content"],
                    memory_type=item["memory_type"], source="extractor",
                    emotional_weight=2.0 if key == "corrections" else 1.0,
                )
