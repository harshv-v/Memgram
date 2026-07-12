"""LoCoMo runner — Memgram on the public long-conversation memory benchmark.

Protocol mirrors bench/quality/run_eval.py: ingest every session, wait for
extraction, then for each QA retrieve top-k memories -> answer from ONLY those
memories -> LLM judge. See README.md in this directory for setup.
"""
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import httpx
from openai import OpenAI

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DATA = Path(__file__).parent / "locomo10.json"
TOPK = int(os.environ.get("EVAL_TOPK", "8"))
N_SAMPLES = int(os.environ.get("EVAL_SAMPLES", "0"))  # 0 = all
ANSWER_MODEL = os.environ.get("EVAL_ANSWER_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o-mini")
CONTRADICTION = os.environ.get("MEMGRAM_EVAL_CONTRADICTION", "0") == "1"
CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain",
            4: "single-hop", 5: "adversarial"}

oai = OpenAI()

_ANSWER_SYS = (
    "You answer a question about a conversation between two people, using ONLY "
    "the provided memories. If the memories do not contain the answer, reply "
    'with exactly "NO_INFO". Be concise — a few words.'
)
_JUDGE_SYS = (
    'You grade a predicted answer against the expected answer. Reply only '
    '"CORRECT" or "WRONG". Ignore casing, pluralization, and extra words; the '
    "prediction is CORRECT if it refers to the same thing. If the expected "
    "answer indicates the question is unanswerable (or expected is NO_INFO/"
    "adversarial), the prediction is CORRECT only if it abstains."
)


class Memgram:
    def __init__(self):
        base = os.environ.get("MEMGRAM_API_BASE_URL", "http://localhost:8000")
        key = os.environ.get("MEMGRAM_API_KEY", "mgram_dev_key")
        self.c = httpx.Client(base_url=base, timeout=120,
                              headers={"Authorization": f"Bearer {key}"})
        self.project = f"locomo-{uuid.uuid4().hex[:6]}"

    def ingest_session(self, user_id: str, turns: list[str]):
        body = {"project_id": self.project, "agent_id": "locomo", "user_id": user_id,
                "messages": [{"role": "user", "content": t} for t in turns],
                "response_text": "Noted."}
        if CONTRADICTION:
            body["contradiction"] = True
        self.c.post("/v1/ingest", json=body).raise_for_status()

    def wait_ready(self, budget: float = 600):
        deadline = time.time() + budget
        while time.time() < deadline:
            if self.c.get("/health").json().get("queue_depth", 1) == 0:
                break
            time.sleep(1)
        time.sleep(2)

    def retrieve(self, user_id: str, q: str, k: int = TOPK) -> list[str]:
        r = self.c.get("/v1/memory/search", params={
            "project_id": self.project, "agent_id": "locomo",
            "user_id": user_id, "query": q, "limit": k})
        r.raise_for_status()
        return [m["content"] for m in r.json()["memories"]]


def sessions_of(conv: dict) -> list[list[str]]:
    """LoCoMo conversation dict -> ordered sessions of speaker-attributed turns."""
    out = []
    i = 1
    while f"session_{i}" in conv:
        sess = conv[f"session_{i}"] or []
        date = conv.get(f"session_{i}_date_time", "")
        turns = [f"[{date}] {t.get('speaker', '?')}: {t.get('text', '')}"
                 for t in sess if t.get("text")]
        if turns:
            out.append(turns)
        i += 1
    return out


def answer_from(memories: list[str], q: str) -> str:
    joined = "\n".join(f"- {m}" for m in memories) or "(no memories)"
    r = oai.chat.completions.create(model=ANSWER_MODEL, temperature=0, messages=[
        {"role": "system", "content": _ANSWER_SYS},
        {"role": "user", "content": f"Memories:\n{joined}\n\nQuestion: {q}\nAnswer:"}])
    return r.choices[0].message.content.strip()


def judge(q: str, expect, pred: str) -> bool:
    r = oai.chat.completions.create(model=JUDGE_MODEL, temperature=0, messages=[
        {"role": "system", "content": _JUDGE_SYS},
        {"role": "user", "content": f"Question: {q}\nExpected: {expect}\nPredicted: {pred}\nGrade:"}])
    return r.choices[0].message.content.strip().upper().startswith("CORRECT")


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY.")
    if not DATA.exists():
        sys.exit(f"Dataset missing: {DATA}\nSee bench/locomo/README.md for download steps.")
    samples = json.loads(DATA.read_text(encoding="utf-8"))
    if N_SAMPLES:
        samples = samples[:N_SAMPLES]

    mg = Memgram()
    by_cat = defaultdict(lambda: [0, 0])
    rows = []
    print(f"LoCoMo | {len(samples)} conversations | top_k={TOPK} "
          f"| contradiction_v2={'ON' if CONTRADICTION else 'off'}")

    for si, sample in enumerate(samples):
        uid = sample.get("sample_id", f"s{si}")
        sess = sessions_of(sample.get("conversation", {}))
        print(f"[{si+1}/{len(samples)}] {uid}: ingesting {len(sess)} sessions ...")
        for turns in sess:
            mg.ingest_session(uid, turns)
        mg.wait_ready()
        qas = sample.get("qa", [])
        for qa in qas:
            q, expect = qa.get("question", ""), qa.get("answer", "NO_INFO")
            cat = CATEGORY.get(qa.get("category"), str(qa.get("category")))
            if cat == "adversarial":
                expect = "NO_INFO"
            pred = answer_from(mg.retrieve(uid, q), q)
            ok = judge(q, expect, pred)
            by_cat[cat][0] += int(ok)
            by_cat[cat][1] += 1
            rows.append({"sample": uid, "category": cat, "q": q,
                         "expect": str(expect), "pred": pred, "ok": ok})

    print("\n" + "=" * 64)
    tot_c = sum(v[0] for v in by_cat.values())
    tot_n = sum(v[1] for v in by_cat.values())
    for cat, (c, n) in sorted(by_cat.items()):
        print(f"{cat:<14}{c:>5}/{n:<5} = {c/n*100:5.1f}%")
    print("-" * 64)
    print(f"{'OVERALL':<14}{tot_c:>5}/{tot_n:<5} = {tot_c/tot_n*100:5.1f}%")
    out = Path(__file__).parent / "results_locomo.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nPer-question detail -> {out}")


if __name__ == "__main__":
    main()
