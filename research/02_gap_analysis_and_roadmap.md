# Gap Analysis & Research-Grounded Roadmap

What Memgram is missing vs the literature + leading systems, prioritized, with a
concrete redesign for the #1 gap (contradiction) grounded in the Mem0 paper.

---

## The gaps, ranked by ROI

| # | Gap | Evidence | Source of the fix |
|---|---|---|---|
| 1 | **No UPDATE/DELETE memory operation** (contradiction) | eval `update` axis 80%; our heuristic regressed it to 20% | Mem0 ADD/UPDATE/DELETE/NOOP; MemGPT core_memory_replace |
| 2 | **No real benchmark** | only a 24-Q homemade eval | LOCOMO / LongMemEval (papers downloaded) |
| 3 | **No multi-hop / graph reasoning** | independent vectors only | Mem0g graph; MemGPT heartbeats; HippoRAG |
| 4 | **Summarizer is async, not pressure-time** | over-budget turn still fires bloated | MemGPT memory-pressure flush |
| 5 | **No importance/salience at write** | every fact equal (except corrections) | Generative Agents importance score |
| 6 | **Single-stage retrieval** | top-k, no re-rank/decompose | MemGPT multi-step retrieval |

**Memgram's moat to protect:** reinforcement + Ebbinghaus decay (active forgetting).
Neither Mem0 nor MemGPT models memory *strength/decay over time*. The strategy is
**keep the moat, close gap #1 and #2.**

---

## Gap #1 redesign — contradiction as an OPERATION, not an archival hunt

### Why my first attempt regressed (root cause, precisely)
I asked the LLM: *"given these NEW facts and these EXISTING memories, which existing
ones are obsolete (old_id, by)?"* Problems:
- **Unbounded blame assignment** — the model picks arbitrary old rows to kill; with
  384-d embeddings clustering tightly, candidate sets are noisy → it archived correct
  facts (even archived the NEW fact).
- **Wrong unit of decision** — it reasons over a *set* relationship instead of a
  clean per-fact integration choice.

### The Mem0-grounded design (operation selection)
Mirror Mem0's update phase exactly, because it's proven on LOCOMO:

For **each newly extracted candidate fact ω**:
1. Retrieve its **top-s (≈5–10) most similar active memories** (not "all near").
2. One LLM **function/tool call** returns a single operation **on ω**:
   - `ADD` → store ω (no equivalent exists).
   - `UPDATE(target_id)` → ω augments an existing memory; merge/replace its content,
     keep the id (preserve reinforcement history).
   - `DELETE(target_id)` → ω contradicts `target_id`; **soft-invalidate** it
     (`superseded_by = ω.id`, tier=archived — infra already built).
   - `NOOP` → ω adds nothing; drop it.
3. The decision is **local to ω and its own neighbours** — bounded, structured,
   one target. This is the key difference from my version.

### Concrete differences from my failed code
| My (regressed) version | Mem0-grounded version |
|---|---|
| "which OLD ids are obsolete?" | "what OP for THIS new fact?" (ADD/UPDATE/DELETE/NOOP) |
| candidates = all within dist<0.45 | top-s nearest only |
| could archive new/unrelated facts | op acts on ω; DELETE targets one specific neighbour |
| no merge path | UPDATE merges, preserving reinforcement_count |
| ran always when candidates existed | NOOP is a first-class, common outcome |

### Soft-invalidate, keep history (both Mem0g and our infra agree)
Mark contradicted memories invalid, **don't physically delete** — enables "where did
they live *before*?" temporal queries. Our `superseded_by` column already does this;
retrieval + decay already skip such rows. So the infra is right; only the
*decision algorithm* changes.

### Test plan (so we don't regress again)
- Re-introduce behind `MEMGRAM_CONTRADICTION=1`, default off until it **beats** the
  no-contradiction baseline (96% overall / 80% update) on our eval.
- Add explicit temporal cases (Berlin→Munich, Rust→Go, role change, deploy-day).
- Adopt Mem0's eval rigor: **average ≥5 runs** (J is stochastic — our ±4% swings are
  run noise, not signal).

---

## Gap #2 — real benchmark (do this *alongside* #1)
- Wire **LOCOMO** (10 convos, ~200 Q, categories: single/multi-hop/temporal/open) and
  **LongMemEval** through the existing `bench/quality` adapter interface.
- Report per-category + mean ± std over ≥5 runs, vs Mem0's published numbers.
- This converts "we tie on 24 questions" into a defensible, comparable result, and
  gives contradiction work a real temporal-reasoning score to move.

---

## Gaps #3–6 (after #1–2 land)
- **#4 synchronous summarizer** (MemGPT memory-pressure): cheapest high-value item —
  compress *before* the over-budget turn fires, not after. Matches the original spec.
- **#5 importance scoring** (Generative Agents: retrieval = recency × importance ×
  relevance; we have retention × emotion × relevance — add an LLM importance score at
  write so trivia decays faster than identity).
- **#3 multi-hop / graph**: AGE is provisioned; Mem0g shows the entity-graph + update-
  resolver pattern. Larger effort; gate on whether the benchmark shows multi-hop loss.
- **#6 iterative retrieval** (MemGPT heartbeats): query decomposition / re-rank.

---

## Immediate next step
Rebuild contradiction resolution with the **operation-selection (ADD/UPDATE/DELETE/
NOOP)** design, validate it **beats** the baseline on the eval (with ≥5-run averaging)
before enabling, and stand up a real LOCOMO run to measure temporal reasoning properly.
See [01_competitors_mem0_memgpt.md](01_competitors_mem0_memgpt.md) and
[00_memgram_deep_dive.md](00_memgram_deep_dive.md).
