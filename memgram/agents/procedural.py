"""Procedural agent — learns from tool usage. Reads a conversation's tool_call /
tool_result turns and distills reusable lessons ("calling X without Y times out;
include Y") into procedural memory (memory_type='procedure'). Dedup + reinforcement
do the rest: a pattern that recurs gets reinforced, so repeated tool behavior
hardens into a strong procedural memory the assembler can surface next time.

Opt-in: only runs when the `procedural` feature is enabled (e.g. the `coding`
preset). Procedural memories get higher stability (slow decay) since tool
behavior changes less often than facts."""
from memgram.agents.base import BaseAgent, fast_model

_SYSTEM = """You analyze an AI agent's tool usage in a conversation and extract reusable PROCEDURAL lessons.
Look at the tool_call and tool_result turns. Return ONLY a JSON object:
{"procedures": [{"content": "..."}]}
Each item is one general, reusable rule about using a tool — which tool, what input pattern, and whether it tends to succeed or fail — phrased for next time.
Good: "Calling the search API without a date filter returns a timeout; always include a date range."
Good: "The deploy tool succeeds when run after the test suite passes."
Rules:
- Only tool-related lessons. If no tool was used or there's nothing reusable, return an empty array.
- Each item is a single self-contained sentence. Quality over quantity."""


class ProceduralAgent(BaseAgent):
    def __init__(self, store, config=None, llm=None):
        super().__init__(store, config, llm)
        if not (config or {}).get("model"):
            self.model = fast_model()

    def build_prompt(self, job: dict) -> list[dict]:
        convo = "\n".join(
            f"{m['role']}: {m['content']}" for m in job["messages"]
            if m.get("role") in ("user", "assistant", "tool_call", "tool_result")
            and isinstance(m.get("content"), str)
        )
        if job.get("response_text"):
            convo += f"\nassistant: {job['response_text']}"
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Conversation with tool usage:\n{convo}"},
        ]

    def parse_output(self, raw: str) -> list[dict]:
        data = self.parse_json(raw)
        return [{"content": i["content"]} for i in data.get("procedures", []) if i.get("content")]

    async def on_success(self, job: dict, result: list) -> None:
        for item in result:
            await self.store.upsert_semantic(
                project_id=job["project_id"], agent_id=job["agent_id"],
                user_id=job["user_id"], content=item["content"],
                memory_type="procedure", source="procedural",
                stability=3.0, provenance="tool usage",
            )
