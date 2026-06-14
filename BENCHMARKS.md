# Memgram Benchmarks

Reproducible latency, scaling, and head-to-head numbers for Memgram. All runs are
single-client on one machine against the local Docker stack (Postgres/pgvector +
Valkey), real OpenAI `gpt-4o-mini` (extraction) and `text-embedding-3-small`
(embeddings). These are **engineering numbers, not marketing** — read the caveats.

> Reproduce: `OPENAI_API_KEY=sk-... python bench/bench.py`
> Head-to-head: `OPENAI_API_KEY=sk-... python bench/compare_mem0.py`

---

## 1. Hot-path overhead (what the SDK adds before the LLM call)

Memgram does *not* benchmark the LLM completion itself (that's the provider's
latency). It measures exactly what it inserts into the request path.

| Component | p50 | p95 | p99 |
|---|---|---|---|
| Instruction fetch (Valkey-cached) | **1.9 ms** | 2.3 ms | 3.0 ms |
| Semantic search — pgvector only (embedding cached) | 45.8 ms | 60 ms | 73 ms |
| Semantic search — full (OpenAI embedding + pgvector) | **346 ms** | 521 ms | 1025 ms |
| Ingest / post-hook ack (202, fire-and-forget) | 51.9 ms | 56.7 ms | — |

**Takeaway:** the **embedding round-trip to OpenAI (~300 ms) dominates** the hot
path. Memgram's own infrastructure (Valkey cache + pgvector) is the small part —
instruction injection is effectively free at ~2 ms. The single highest-leverage
optimization is a shared embedding cache or a local embedding model (the
`MEMGRAM_LLM_BASE_URL` / `MEMGRAM_EMBED_MODEL` escape hatch already exists).

## 2. Search scaling (pgvector HNSW)

Corpus grown with synthetic unit vectors; search measured at each size.

| Corpus | pgvector-only (cached embedding) p50 / p95 | full (embedding + pgvector) p50 / p95 |
|---|---|---|
| 100 | 7.3 / 9.3 ms | 317 / 497 ms |
| 1,000 | 11.9 / 14.9 ms | 314 / 530 ms |
| 10,000 | 44.7 / 56.5 ms | 354 / 517 ms |

**Takeaway:** HNSW scales sub-linearly — 100× more rows (100 → 10k) only ~6× the
DB-side latency (7 → 45 ms). The full-query number barely moves because it's
embedding-bound, not search-bound.

## 3. Head-to-head: Memgram vs Mem0

Same models, same 8 labeled conversations. **Mem0 `add()` is synchronous**
(extraction inline) while **Memgram ingest is async** (background worker), so the
write path is reported asymmetrically — comparing Mem0's add to Memgram's ack
alone would be misleading.

| Axis | Mem0 | Memgram |
|---|---|---|
| **Write (sync add)** | p50 **2253 ms** / p95 3292 ms — queryable immediately | — |
| **Write (async ack)** | — | p50 **59 ms** / p95 69 ms — queryable after ~18 s (worker) |
| **Search latency** | p50 413 ms / p95 596 ms | p50 **7.6 ms** (cached) / p95 899 ms (cold) |
| **Extraction recall** | 8 / 8 | 8 / 8 |
| **Stored memories** | 8 | 9 |

**Honest reading:**

- **Write path is a philosophy difference, not a win.** Mem0 blocks the caller
  ~2.3 s to extract, but the memory is instantly searchable. Memgram returns in
  ~59 ms and extracts in the background — far better *user-facing* latency, at the
  cost of a freshness lag. With the concurrent worker (8 consumers) that lag is
  **~4.7 s** for 8 facts; it was ~18 s when the worker processed jobs serially.
- **Search:** for *repeated* queries Memgram's in-process embedding cache + pgvector
  gives sub-10 ms; for *novel* queries both systems are embedding-bound (~300–400 ms,
  see §1). Mem0 does not appear to cache identical queries in this run.
- **For real quality, see §4** — this section is latency only; the keyword probe
  here can't differentiate quality.

## 4. Quality: head-to-head recall (multi-session eval)

A LOCOMO-style eval (`bench/quality/`): 4 personas, multi-session conversations,
24 questions. Every system gets the same conversations; the pipeline per question
is identical — **retrieve top-k → answer using only those memories → LLM-judge** —
so the only variable measured is *what each system remembered*. Question types:
`single` (stated once), `update` (a fact changed; latest should win), `preference`,
and `absent` (never stated → the system must **abstain**; this is the hallucination
test). Same models for both (`gpt-4o-mini` + `text-embedding-3-small`).

| System | Overall | single | update | preference | absent (no-hallucination) |
|---|---|---|---|---|---|
| **Memgram** | **96%** (23/24) | 100% | 80% | 100% | 100% |
| **Mem0** | **96%** (23/24) | 100% | 80% | 100% | 100% |

**Honest reading:**

- **Memgram is at parity with Mem0** on this eval — including a perfect
  hallucination score (both correctly abstain on all `absent` questions).
- **Both share one weakness: temporal updates.** The single miss for each is the
  same — a user promoted from "backend engineer" to "tech lead"; both systems
  retrieve the *stale* role. Similarity-based memory reinforces/duplicates but
  doesn't *supersede* a contradicted fact. This is the concrete motivation for
  contradiction-resolution / memory versioning (see the roadmap).
- **This is a small set (24 Q).** It's enough to show parity and to surface the
  temporal weakness, not enough for a leaderboard claim — run full LOCOMO /
  LongMemEval through the same harness (the dataset shape is compatible) to scale it.

> Reproduce: `OPENAI_API_KEY=sk-... python bench/quality/run_eval.py`
> Per-question detail is written to `bench/quality/results.json`.

### Systems compared (and why some are deferred)
- **Mem0** — clean memory API (`add`/`search`); a true drop-in adapter, included.
- **Letta (MemGPT) / Zep** — *agent/service* frameworks: memory lives inside an
  agent loop or a separate server, so there's no equivalent "ingest → retrieve"
  call and each needs its own running service (and an isolated env to avoid
  dependency conflicts). The harness is pluggable (`bench/quality/adapters.py`,
  `ADAPTERS`); adding them is a contained task but out of scope for this first,
  apples-to-apples pass. Documented rather than faked.

## 5. Durability (at-least-once delivery)

The job queue uses Valkey **Streams + consumer groups**. Verified live: a job
stranded in a crashed worker's pending list (delivered, never acked) is reclaimed
via `XAUTOCLAIM` by another worker and completed — nothing is lost on a crash.
Jobs that fail past `max_deliveries` go to a dead-letter stream (`memgram:jobs:dead`,
surfaced in `/health`). This is a correctness property Mem0's synchronous model
doesn't need (and doesn't have a queue for).

---

## Caveats (apply to every number above)

- Single client, single machine, local loopback — **not** a throughput or
  distributed benchmark. (Open-loop load testing with k6/vegeta is future work.)
- Storage backends differ by design: Mem0 → local Qdrant; Memgram → pgvector over
  the Docker API (a localhost network hop).
- Embedding latency depends on OpenAI and your network; the ~300 ms is the single
  biggest, most variable term.
- `gpt-4o-mini` extraction; quality differs from `gpt-4o`. Recall here is a small
  keyword-grounded probe, not a published memory benchmark.
