"""Quality eval: head-to-head memory recall for Memgram vs Mem0 (and others).

For each persona we feed every system the same multi-session conversation, then
ask questions whose answers must be recalled from memory. The pipeline per
question is identical across systems:

    retrieve top-k memories  ->  answer using ONLY those memories  ->  LLM-judge

so the only thing being measured is the quality of what each system remembered.
"absent" questions (answer = NO_INFO) reward ABSTAINING — they measure resistance
to hallucination, which for a memory layer matters as much as recall.

Run (Memgram stack up; `pip install mem0ai`):
    OPENAI_API_KEY=sk-...  python bench/quality/run_eval.py
    OPENAI_API_KEY=sk-...  EVAL_SYSTEMS=memgram,mem0 python bench/quality/run_eval.py
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from adapters import ADAPTERS  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o-mini")
ANSWER_MODEL = os.environ.get("EVAL_ANSWER_MODEL", "gpt-4o-mini")
TOPK = int(os.environ.get("EVAL_TOPK", "6"))
oai = OpenAI()

_ANSWER_SYS = (
    "You answer a question using ONLY the provided memories about the user. "
    'If the memories do not contain the answer, reply with exactly "NO_INFO". '
    "Be concise — a few words."
)
_JUDGE_SYS = (
    "You grade whether a predicted answer matches the expected answer for a "
    'question. Reply with only "CORRECT" or "WRONG".\n'
    "Treat the prediction as CORRECT if it refers to the same thing as the "
    "expected answer, ignoring pluralization, casing, punctuation, and extra "
    'words (e.g. "Tuesdays" matches "Tuesday"; "Munich, Germany" matches '
    '"Munich"). It is WRONG if it states a different or outdated fact.\n'
    'If the expected answer is "NO_INFO", the prediction is CORRECT only if it '
    "abstains (says it doesn't know / not mentioned / NO_INFO), and WRONG if it "
    "states any specific fact."
)


def answer_from(memories, q):
    joined = "\n".join(f"- {m}" for m in memories) or "(no memories)"
    r = oai.chat.completions.create(model=ANSWER_MODEL, temperature=0, messages=[
        {"role": "system", "content": _ANSWER_SYS},
        {"role": "user", "content": f"Memories:\n{joined}\n\nQuestion: {q}\nAnswer:"}])
    return r.choices[0].message.content.strip()


def judge(q, expect, pred):
    r = oai.chat.completions.create(model=JUDGE_MODEL, temperature=0, messages=[
        {"role": "system", "content": _JUDGE_SYS},
        {"role": "user", "content": f"Question: {q}\nExpected: {expect}\nPredicted: {pred}\nGrade:"}])
    return r.choices[0].message.content.strip().upper().startswith("CORRECT")


def run_system(sys_name, cases):
    adapter = ADAPTERS[sys_name]()
    by_type = defaultdict(lambda: [0, 0])  # type -> [correct, total]
    rows = []
    for case in cases:
        persona = case["persona"]
        adapter.setup(persona)
        for turns in case["sessions"]:
            adapter.ingest(persona, turns)
        adapter.wait_ready()
        for ques in case["questions"]:
            mems = adapter.retrieve(persona, ques["q"], TOPK)
            pred = answer_from(mems, ques["q"])
            ok = judge(ques["q"], ques["expect"], pred)
            by_type[ques["type"]][0] += int(ok)
            by_type[ques["type"]][1] += 1
            rows.append((persona, ques["type"], ques["q"], ques["expect"], pred, ok))
    return by_type, rows


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY.")
    data = json.loads((Path(__file__).parent / "dataset.json").read_text(encoding="utf-8"))
    cases = data["cases"]
    systems = os.environ.get("EVAL_SYSTEMS", "memgram,mem0").split(",")
    nq = sum(len(c["questions"]) for c in cases)
    print("=" * 72)
    print(f"MEMGRAM QUALITY EVAL  |  {len(cases)} personas, {nq} questions, top_k={TOPK}")
    print("=" * 72)

    results = {}
    for s in systems:
        s = s.strip()
        if s not in ADAPTERS:
            print(f"  (skip unknown system: {s})")
            continue
        print(f"\nRunning {s} ...")
        try:
            results[s] = run_system(s, cases)
        except Exception as e:
            print(f"  {s} FAILED: {e}")

    types = ["single", "update", "preference", "absent"]
    print("\n" + "=" * 72)
    print(f"{'system':<10}{'overall':>9}  " + "  ".join(f"{t:>10}" for t in types))
    print("-" * 72)
    for s, (by_type, _rows) in results.items():
        tot_c = sum(v[0] for v in by_type.values())
        tot_n = sum(v[1] for v in by_type.values())
        cells = []
        for t in types:
            c, n = by_type.get(t, [0, 0])
            cells.append(f"{(c/n*100 if n else 0):>9.0f}%" if n else f"{'-':>10}")
        print(f"{s:<10}{tot_c/tot_n*100:>8.0f}%  " + "  ".join(cells)
              + f"   ({tot_c}/{tot_n})")
    print("\nLegend: single=stated once · update=fact changed (latest wins) · "
          "preference · absent=never stated (must abstain; hallucination test)")

    # dump per-question detail for the report
    out = {s: [{"persona": p, "type": t, "q": q, "expect": e, "pred": pr, "ok": ok}
               for (p, t, q, e, pr, ok) in rows] for s, (_, rows) in results.items()}
    (Path(__file__).parent / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nPer-question detail -> bench/quality/results.json")


if __name__ == "__main__":
    main()
