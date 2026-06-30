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
from memgram.prompts import get_prompt

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

_CONTRA_SYSTEM = """You detect when a NEW fact about a user makes an EXISTING memory obsolete.
A memory is obsolete ONLY when a new fact updates the SAME attribute to a different value —
the user moved to a new city, changed jobs or title, switched a tool or stack, changed a habit.
Do NOT mark memories that are still true, merely related, or just additional detail.
Return ONLY JSON: {"superseded": [{"old_id": "<id of the now-obsolete memory>", "by": <index of the new fact that replaces it>}]}.
Return an empty list if nothing is truly obsolete."""


class ExtractorAgent(BaseAgent):
    def __init__(self, store, config=None, llm=None):
        super().__init__(store, config, llm)
        cfg = config or {}
        if not cfg.get("model"):
            self.model = fast_model()
        self.faithful = cfg.get(
            "faithfulness", os.environ.get("MEMGRAM_FAITHFULNESS", "1") != "0")
        # Experimental + OFF by default: the naive supersession heuristic regressed
        # the eval's `update` axis (80% -> 20%) by over-archiving correct facts.
        # Pending a research-driven redesign (see research/). Enable with
        # MEMGRAM_CONTRADICTION=1 only for experiments.
        self.contradiction = cfg.get(
            "contradiction", os.environ.get("MEMGRAM_CONTRADICTION", "0") != "0")

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
            {"role": "system", "content": get_prompt("extractor.system", _SYSTEM)},
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
            {"role": "system", "content": get_prompt("extractor.verify", _VERIFY_SYSTEM)},
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
        stored: list[tuple[str, str]] = []  # (content, new_id) for the contradiction pass
        for idx, (key, item) in enumerate(candidates):
            if idx not in keep:
                logger.info("extractor: dropped unsupported memory: %s", item["content"])
                continue
            res = await self.store.upsert_semantic(
                project_id=job["project_id"], agent_id=job["agent_id"],
                user_id=job["user_id"], content=item["content"],
                memory_type=item["memory_type"], source="extractor",
                emotional_weight=2.0 if key == "corrections" else 1.0,
                provenance=provenance,
            )
            stored.append((item["content"], res["id"]))

        if self.contradiction and self._llm is not None and stored:
            await self._resolve_contradictions(job, stored)

    async def _resolve_contradictions(self, job: dict, stored: list[tuple[str, str]]) -> None:
        """For each freshly stored fact, find near-but-not-duplicate existing
        memories and, if the LLM judges them obsolete, supersede (archive) them.
        Only fires when a candidate exists — no candidate, no extra LLM call."""
        # Only PRIOR memories can be superseded — never a fact from this same turn
        # (all of this turn's facts are already stored/active, so without this a new
        # fact could be wrongly archived as another new fact's "old" candidate).
        batch_ids = {new_id for _, new_id in stored}
        cand_by_new: dict[int, list[dict]] = {}
        for i, (content, new_id) in enumerate(stored):
            cands = await self.store.find_similar_active(
                job["project_id"], job["agent_id"], job["user_id"], content)
            cands = [c for c in cands if c["id"] not in batch_ids]
            if cands:
                cand_by_new[i] = cands
        if not cand_by_new:
            return
        try:
            new_block = "\n".join(f"{i}. {stored[i][0]}" for i in cand_by_new)
            seen: dict[str, str] = {}
            for cands in cand_by_new.values():
                for c in cands:
                    seen[c["id"]] = c["content"]
            cand_block = "\n".join(f"- id={cid}: {txt}" for cid, txt in seen.items())
            raw = await self._call_llm([
                {"role": "system", "content": get_prompt("extractor.contradiction", _CONTRA_SYSTEM)},
                {"role": "user", "content":
                    f"NEW FACTS:\n{new_block}\n\nEXISTING MEMORIES:\n{cand_block}\n\nReturn the JSON."},
            ])
            for s in self.parse_json(raw).get("superseded", []):
                old_id, by = s.get("old_id"), s.get("by")
                if old_id is None or by is None:
                    continue
                try:
                    new_id = stored[int(by)][1]
                except (TypeError, ValueError, IndexError):
                    continue
                if old_id != new_id:
                    await self.store.supersede(old_id, new_id)
                    logger.info("superseded stale memory %s (replaced by %s)", old_id, new_id)
        except Exception as e:  # supersession is best-effort; never break extraction
            logger.warning("contradiction check failed (%s); skipping", e)
