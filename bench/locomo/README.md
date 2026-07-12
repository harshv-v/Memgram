# LoCoMo — public long-conversation memory benchmark

[LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., 2024 — the
benchmark Mem0 quotes in its papers/blog) tests memory over very long multi-
session conversations. Running Memgram on it gives us a **publicly comparable**
number instead of only our in-house eval.

## Get the dataset

```bash
git clone https://github.com/snap-research/locomo /tmp/locomo
cp /tmp/locomo/data/locomo10.json bench/locomo/
```

## Run (Memgram stack must be up)

```bash
OPENAI_API_KEY=sk-...  python bench/locomo/run_locomo.py                 # default settings
EVAL_SAMPLES=3         python bench/locomo/run_locomo.py                 # quick subset
MEMGRAM_EVAL_CONTRADICTION=1 python bench/locomo/run_locomo.py           # with v2 integration
```

Scores print per category. Category labels in the dataset (per the LoCoMo
paper): 1 = multi-hop, 2 = temporal, 3 = open-domain, 4 = single-hop,
5 = adversarial (correct answer = abstain). Verify against the dataset README
if scores look misaligned.

Notes:
- The answer + judge stage is the same protocol as `bench/quality/run_eval.py`
  (answer from retrieved memories only; LLM judge), so numbers are comparable
  across our two benches.
- Full locomo10 is ~10 conversations × ~200 QA — budget ~$2-4 of gpt-4o-mini
  and 30-60 min including extraction. Use EVAL_SAMPLES for smoke runs.
