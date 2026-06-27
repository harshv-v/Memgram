# Memgram — Deep Dive on the Current System (baseline for gap analysis)

Honest, in-depth snapshot of what Memgram *actually does today* (verified this
build), so the gap analysis vs Mem0 / MemGPT is grounded in reality, not the pitch.

---

## 1. The memory model (four pillars)

| Pillar | Table / type | Written by | Decay | Injected into prompt? |
|---|---|---|---|---|
| **Instruction** | `instructions` | user (active) / agent (pending) | never | yes — always, first, priority-ordered |
| **Semantic** | `semantic_memories` | extractor / reflection | Ebbinghaus (days) | yes — top-k by rank |
| **Episodic** | `episodic_logs` | ingest (raw turns + tool turns) | fast (hours) | no (raw); summarized when over budget |
| **Procedural** | `semantic_memories` w/ `memory_type='procedure'` | procedural agent (tool outcomes) | slow (weeks) | yes — via the same semantic search (opt-in: `coding` preset) |

Embeddings: **local fastembed `bge-small-en-v1.5` (384d)** in-process by default
(`MEMGRAM_EMBEDDER=local`), OpenAI 1536d optional. Vector index: pgvector **HNSW**.

## 2. The agents (what each is, exactly)

All agents = a prompt + a JSON response + a retry loop (`BaseAgent`, ~60 lines). No
framework. Two model tiers: `fast_model` (gpt-4o-mini) for extraction/summarization,
`quality_model` (gpt-4o) for reflection/proposals.

1. **Extractor** (`agents/extractor.py`) — every completed turn. Distills
   facts/preferences/entities/corrections → semantic memory. Has a **faithfulness
   gate** (a 2nd LLM "fact-checker" call) that drops anything not supported by the
   transcript; stores a provenance excerpt. *(Contains an experimental contradiction
   step — OFF by default, it regressed the eval; see §6.)*
2. **Reflection** (`agents/reflection.py`) — every N interactions / 24h sweep. Reads
   unreflected episodic logs, distills durable insights, flags habit candidates
   (reinforcement ≥ 7).
3. **Proposer** (`agents/proposer.py`) — on a habit candidate. Generates a
   natural-language instruction with `status='pending'` (the trust gate — agents
   never write active instructions). Similarity-checks against existing instructions.
4. **Decay** (`agents/decay.py`) — nightly, **pure SQL, $0**. `R = exp(-t/S)`,
   re-tiers active→fading→archived; flags stale agent-proposed instructions for
   review. Skips `promoted` and `superseded` rows.
5. **Summarizer** (`agents/summarizer.py`) — when a turn exceeds 60% of the context
   limit. Compresses the conversation into a structured session summary. **Runs
   async (enqueued), NOT synchronously** — a documented tradeoff; the over-budget
   prompt still fires with bloated context that turn.
6. **Procedural** (`agents/procedural.py`) — when a completed turn used tools.
   Distills tool success/failure lessons into procedure memory. Opt-in.

## 3. How memory is written (the dedup→reinforce core)

`store.upsert_semantic` (advisory-locked per user for concurrency safety):
- embed content → nearest active memory.
- if cosine distance **< 0.15** → **reinforce** (count++, stability ×(1+0.2·R),
  retention=1.0, restore tier) — *no new row*.
- else → INSERT.
This is how "the same fact said 50 times" becomes one row with `reinforcement_count=50`,
not 50 rows. **This is the only contradiction-relevant mechanism that works today**
(it collapses *near-identical* repeats; it does NOT handle *changed* facts).

## 4. How memory is read (the hot path, ranked)

`retriever.search`: embed query → pgvector HNSW search, ranked by
**`cosine_similarity × retention_score × emotional_weight`**, excluding archived /
superseded, scope-filtered (RLS sets `app.current_user_id`). Accessed memories are
**reinforced on read** (access = rehearsal). Then the assembler injects:
1. instruction block (Valkey-cached, ~2ms),
2. developer system prompt,
3. semantic block (top-k),
…then the conversation. Memory failures fall through — never break the LLM call.

## 5. Verified performance (this build, measured)

- Hot-path search: **~38ms** at 10k memories with local embedder (was 346ms with
  OpenAI's network hop). Instruction fetch ~2ms.
- HNSW scaling: 7ms@100 → 45ms@10k (sub-linear).
- Durability: at-least-once via Valkey Streams; crashed-worker jobs reclaimed.
- Concurrency: 8 consumers; time-to-queryable ~4.7s for 8 facts.
- Quality eval (24Q multi-session, vs Mem0, same models): **96% (23/24)**, tied
  with Mem0. Perfect on single-fact / preference / hallucination-abstention.

## 6. The hard gaps (honest, evidence-based)

1. **Temporal / contradiction resolution — UNSOLVED and the #1 quality gap.**
   Cosine similarity retrieves *both* "lives in Berlin" and "lives in Munich"; a
   higher reinforcement_count on the stale one can outrank the new reality. The
   eval's `update` axis sits at **80%** with no special handling.
   - My naive fix (LLM picks what to archive) **regressed it to 20-40%** by
     over-archiving correct facts. Disabled by default. **This is the central
     research question** — how do the published systems actually do it?
2. **Summarizer is async, not synchronous** — doesn't prevent the over-budget turn.
3. **No graph / multi-hop reasoning** — facts are independent vectors; "who is my
   manager's manager" style queries aren't supported (no entity graph queries,
   though AGE is provisioned).
4. **Retrieval is single-stage top-k** — no re-ranking, no query decomposition, no
   iterative/agentic retrieval (MemGPT-style self-paging).
5. **No importance/salience scoring at write** — every extracted fact is equal
   (emotional_weight only bumps corrections). Generative-Agents-style importance
   scoring is absent.
6. **Episodic growth unbounded** — fast-decay but no TimescaleDB archival; a problem
   only at scale.
7. **Quality eval is tiny (24Q)** — proves parity + the temporal gap, not a
   leaderboard claim. Need LOCOMO / LongMemEval for real numbers.

## 7. What to learn from the literature (drives the next docs)

- **MemGPT**: OS-style tiered memory + self-editing (the agent decides what to page
  in/out, and *edits/overwrites* its own memory) — directly relevant to contradiction.
- **Generative Agents**: importance scoring + reflection trees + retrieval =
  recency × importance × relevance (we have relevance × retention × emotion — what
  are we missing?).
- **Mem0**: explicit ADD/UPDATE/DELETE memory operations with an LLM deciding the op
  (this is *exactly* the contradiction mechanism we lack), `update()` with timestamps.
- **LOCOMO / LongMemEval**: the benchmarks; categories incl. temporal reasoning.
- **A-MEM / agentic memory**: dynamic memory organization, linking, evolution.
