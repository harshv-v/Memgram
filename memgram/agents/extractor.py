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

_OP_SYSTEM = """You integrate ONE new fact about a user into their memory store.
Given the NEW FACT and its most similar EXISTING MEMORIES, choose exactly one operation:
- "ADD"    — the fact is new information no existing memory covers.
- "UPDATE" — an existing memory describes the SAME thing and the fact refines or adds detail
             to it; rewrite that memory to a single merged sentence. Set target_id and content.
- "DELETE" — the fact makes an existing memory UNTRUE (same attribute, different value: moved
             city, changed job or title, switched tool or stack, changed a habit, or an explicit
             negation like "no longer"). Set target_id to the now-false memory.
- "NOOP"   — the fact adds nothing beyond what existing memories already say.
Rules:
- Choose UPDATE or DELETE ONLY when an existing memory states the same attribute of the same
  subject. Merely related or additional facts are ADD. When unsure, choose ADD.
- target_id MUST be copied exactly from the EXISTING MEMORIES list.
Return ONLY JSON: {"op": "ADD" | "UPDATE" | "DELETE" | "NOOP", "target_id": "<id or null>", "content": "<merged single sentence for UPDATE, else null>"}"""


class ExtractorAgent(BaseAgent):
    def __init__(self, store, config=None, llm=None):
        super().__init__(store, config, llm)
        cfg = config or {}
        if not cfg.get("model"):
            self.model = fast_model()
        self.faithful = cfg.get(
            "faithfulness", os.environ.get("MEMGRAM_FAITHFULNESS", "1") != "0")
        # Contradiction v2 — Mem0-grounded OPERATION SELECTION (ADD/UPDATE/DELETE/
        # NOOP), one bounded decision per new fact against its top-s neighbours.
        # (v1 asked "which old ids are obsolete?" over a noisy candidate set and
        # regressed the eval's update axis 80%→20%; see research/02.) Still OFF by
        # default until it beats the no-contradiction baseline on the eval.
        # Enable with MEMGRAM_CONTRADICTION=1.
        self.contradiction = cfg.get(
            "contradiction", os.environ.get("MEMGRAM_CONTRADICTION", "0") != "0")
        self.op_top_s = int(cfg.get(
            "contradiction_top_s", os.environ.get("MEMGRAM_CONTRADICTION_TOP_S", "8")))

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
        # Ids written this turn — this turn's facts must never be UPDATE/DELETE
        # targets for each other (they're all simultaneously true).
        batch_ids: set[str] = set()
        for idx, (key, item) in enumerate(candidates):
            if idx not in keep:
                logger.info("extractor: dropped unsupported memory: %s", item["content"])
                continue
            store_kw = dict(
                project_id=job["project_id"], agent_id=job["agent_id"],
                user_id=job["user_id"], content=item["content"],
                memory_type=item["memory_type"], source="extractor",
                emotional_weight=2.0 if key == "corrections" else 1.0,
                provenance=provenance,
            )
            if self.contradiction and self._llm is not None:
                await self._integrate(job, store_kw, batch_ids)
            else:
                res = await self.store.upsert_semantic(**store_kw)
                batch_ids.add(res["id"])

    async def _integrate(self, job: dict, store_kw: dict, batch_ids: set[str]) -> None:
        """Operation-selection integration (Mem0-grounded): for ONE new fact ω,
        retrieve its top-s nearest active memories and have the LLM pick a single
        bounded operation on ω — ADD / UPDATE(target) / DELETE(target) / NOOP.
        The decision is local to ω and its own neighbours: one fact, one op, at
        most one target, and the target must come from the candidate list. Any
        failure degrades to ADD — integration must never lose a real fact."""
        content = store_kw["content"]
        cands = await self.store.find_similar_active(
            job["project_id"], job["agent_id"], job["user_id"], content,
            limit=self.op_top_s)
        cands = [c for c in cands if c["id"] not in batch_ids]
        if not cands:  # nothing comparable exists -> ADD, no extra LLM call
            res = await self.store.upsert_semantic(**store_kw)
            batch_ids.add(res["id"])
            return

        op, target_id, merged = "ADD", None, None
        try:
            cand_block = "\n".join(f"- id={c['id']}: {c['content']}" for c in cands)
            raw = await self._call_llm([
                {"role": "system", "content": get_prompt("extractor.operation", _OP_SYSTEM)},
                {"role": "user", "content":
                    f"NEW FACT:\n{content}\n\nEXISTING MEMORIES:\n{cand_block}\n\nReturn the JSON."},
            ])
            data = self.parse_json(raw)
            op = str(data.get("op", "ADD")).upper()
            target_id = data.get("target_id")
            merged = data.get("content")
        except Exception as e:  # integration is best-effort; never break extraction
            logger.warning("operation selection failed (%s); defaulting to ADD", e)

        valid_ids = {c["id"] for c in cands}
        if op in ("UPDATE", "DELETE") and target_id not in valid_ids:
            logger.warning("op %s targeted id outside candidate set; defaulting to ADD", op)
            op = "ADD"

        if op == "NOOP":
            logger.info("integration NOOP: %s", content)
            return
        if op == "UPDATE":
            res = await self.store.update_semantic(target_id, merged or content)
            if res is not None:
                logger.info("integration UPDATE %s -> %r", target_id, merged or content)
                batch_ids.add(res["id"])
                return
            op = "ADD"  # target vanished under us -> keep the fact
        res = await self.store.upsert_semantic(**store_kw)
        batch_ids.add(res["id"])
        if op == "DELETE" and res["id"] != target_id:
            await self.store.supersede(target_id, res["id"])
            logger.info("integration DELETE: superseded %s by %s", target_id, res["id"])
