"""Extractor — runs async after every interaction. Pulls facts, preferences,
entities, and corrections out of the conversation and writes them to semantic
memory (with dedup → reinforcement). Corrections get emotional_weight 2.0 so
mistakes persist longer than neutral memories.

Anti-hallucination: a memory system that invents facts is worse than no memory —
a false memory poisons every future prompt. So after extraction we run a strict
faithfulness pass: a second LLM call that keeps only candidates DIRECTLY supported
by the transcript. Each stored memory also keeps a provenance excerpt so it's
auditable. Disable with MEMGRAM_FAITHFULNESS=0 (or config faithfulness=False)."""
import logging
import os

from memgram.agents.base import BaseAgent, fast_model

logger = logging.getLogger("memgram.agents")

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

_VERIFY_SYSTEM = """You are a strict fact-checker for an AI memory system.
Given a conversation and a numbered list of candidate memories extracted from it,
return the indices of candidates that are DIRECTLY STATED or CLEARLY IMPLIED by the
conversation. Reject any candidate not supported by the conversation — a plausible-
sounding but unstated claim is a hallucination and must be rejected.
Return ONLY a JSON object: {"supported": [list of supported indices]}."""


class ExtractorAgent(BaseAgent):
    def __init__(self, store, config=None, llm=None):
        super().__init__(store, config, llm)
        cfg = config or {}
        if not cfg.get("model"):
            self.model = fast_model()
        self.faithful = cfg.get(
            "faithfulness", os.environ.get("MEMGRAM_FAITHFULNESS", "1") != "0")

    def _transcript(self, job: dict) -> str:
        convo = "\n".join(
            f"{m['role']}: {m['content']}" for m in job["messages"]
            if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str)
        )
        if job.get("response_text"):
            convo += f"\nassistant: {job['response_text']}"
        return convo

    def build_prompt(self, job: dict) -> list[dict]:
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Conversation:\n{self._transcript(job)}"},
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

    async def _verify(self, transcript: str, contents: list[str]) -> set[int]:
        """Return the indices of candidates the transcript actually supports."""
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(contents))
        raw = await self._call_llm([
            {"role": "system", "content": _VERIFY_SYSTEM},
            {"role": "user", "content":
                f"Conversation:\n{transcript}\n\nCandidate memories:\n{numbered}\n\n"
                'Return JSON {"supported": [indices that are directly stated or clearly implied]}.'},
        ])
        data = self.parse_json(raw)
        keep = set()
        for i in data.get("supported", []):
            try:
                keep.add(int(i))
            except (TypeError, ValueError):
                pass
        return keep

    async def on_success(self, job: dict, result: dict) -> None:
        candidates = [(key, item) for key, items in result.items() for item in items]
        if not candidates:
            return
        transcript = self._transcript(job)

        keep = set(range(len(candidates)))
        if self.faithful and self._llm is not None:
            try:
                keep = await self._verify(transcript, [c[1]["content"] for c in candidates])
            except Exception as e:  # never let the gate drop real memories on its own error
                logger.warning("faithfulness check failed (%s); storing unverified", e)
                keep = set(range(len(candidates)))

        provenance = transcript[:500]
        for idx, (key, item) in enumerate(candidates):
            if idx not in keep:
                logger.info("extractor: dropped unsupported memory: %s", item["content"])
                continue
            await self.store.upsert_semantic(
                project_id=job["project_id"], agent_id=job["agent_id"],
                user_id=job["user_id"], content=item["content"],
                memory_type=item["memory_type"], source="extractor",
                emotional_weight=2.0 if key == "corrections" else 1.0,
                provenance=provenance,
            )
