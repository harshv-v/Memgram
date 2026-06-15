# Memgram — Handover & Engineering Guide

A practical reference for anyone picking up Memgram: what it is, how it's built,
how to run and verify it, what's hardened, what's deliberately not, and how to
extend it. For the product pitch see [README](README.md); for numbers see
[BENCHMARKS](BENCHMARKS.md).

---

## 1. What it is, in one paragraph

Memgram is a self-hosted memory layer for AI agents. A developer wraps any
OpenAI-compatible client in two lines; thereafter the user's standing
instructions and relevant memories are injected into each prompt automatically,
and the conversation is mined in the background for facts/preferences. Memory is
*active*: it decays when unused, strengthens when reinforced, and repeated
patterns get proposed as permanent instructions a human must approve. The whole
stack is open source except the LLM (OpenAI by default, swappable).

## 2. Architecture at a glance

```
                       HOT PATH (sync, user-facing)
your app ─► mem.wrap(client).chat.completions.create(..., user_id=...)
              │  1. assembler injects: instructions (Valkey-cached) + top-k memories
              │  2. real LLM call, unchanged
              │  3. same response returned
              └► 4. POST /v1/ingest  (fire-and-forget; ~59ms ack)
                       │
        ┌──────────────┴──── BACKGROUND (Valkey Streams → worker) ───────────┐
        │ ingest → episodic log + `extract` job on the stream                  │
        │ worker: K concurrent consumers; ack on success, redeliver on crash   │
        │ extract → (faithfulness gate) → semantic memory (dedup→reinforce)    │
        │ reflect (cadence) → propose (habit) ; nightly decay (pure SQL)       │
        └──────────────────────────────────────────────────────────────────────┘
```

Components (see also the repo map in §8):
- **SDK** (`memgram/sdk`): `Memgram`, the proxy intercept, the context assembler,
  the HTTP client. The only thing on the hot path.
- **API** (`memgram/api`): FastAPI — instructions CRUD, ingest, memory search/list,
  GDPR export/forget, proposals. Connects as a **non-superuser** role so RLS applies.
- **Worker** (`memgram/worker`): Streams-backed queue, the dispatcher (K consumers
  + reclaim loop + dead-letter), and the scheduler (decay/reflection sweeps).
- **Agents** (`memgram/agents`): extractor (+faithfulness), reflection, proposer,
  decay, summarizer. Each is a prompt + JSON + retry loop — no framework.
- **Memory** (`memgram/memory`): store (dedup/reinforce, advisory-lock'd), retriever
  (ranked search + reinforce-on-access), embedder, SQL migrations.
- **Dashboard** (`dashboard/`): Next.js — view/edit instructions, browse memory,
  approve proposals.

## 3. Run it

```bash
cp .env.example .env          # set OPENAI_API_KEY (or MEMGRAM_FAKE_LLM=1)
docker compose up --build     # db (pgvector) + queue (Valkey) + migrate + api + worker
# API on http://localhost:8000  (/health, /docs)

pip install -e .
OPENAI_API_KEY=sk-...  python examples/demo_persistence.py   # persistence in 30s
OPENAI_API_KEY=sk-...  python examples/qa_rag_app.py         # recall + isolation + reinforcement

cd dashboard && npm install && npm run dev                   # http://localhost:3000
```

If ports 5432/6379 are taken locally, set `MEMGRAM_DB_PORT` / `MEMGRAM_QUEUE_PORT`
in `.env` (and match `DATABASE_URL` / `REDIS_URL`).

## 4. Verify it (tests + benchmarks)

```bash
# Tests
MEMGRAM_FAKE_LLM=1 python tests/test_logic.py                                  # 36 pure-logic checks
MEMGRAM_API_KEY=mgram_dev_key python tests/test_thin_slice.py                  # API injection + trust gate
DATABASE_URL=postgresql://memgram:memgram@localhost:5433/memgram \
  MEMGRAM_FAKE_LLM=1 MEMGRAM_FAKE_REDIS=1 python tests/test_pipeline.py        # full pipeline

# Benchmarks
OPENAI_API_KEY=sk-... python bench/bench.py            # hot-path latency + HNSW scaling
OPENAI_API_KEY=sk-... python bench/compare_mem0.py     # latency vs Mem0 (needs `pip install mem0ai`)
OPENAI_API_KEY=sk-... python bench/quality/run_eval.py # recall quality vs Mem0
```

## 5. What's hardened (and proven this far)

| Area | State | Evidence |
|---|---|---|
| Durability | At-least-once via Valkey Streams; crash → job reclaimed; dead-letter stream | verified live (XAUTOCLAIM reclaim) |
| Concurrency | Worker runs K consumers (`MEMGRAM_WORKER_CONCURRENCY`, default 8) | time-to-queryable ~18s → ~4.7s |
| Dedup race | check-then-insert under per-user advisory lock | 12 concurrent identical → 1 row + 11 reinforced |
| Hallucination | extractor faithfulness gate drops unsupported facts; provenance stored | kept real facts, dropped invented ones |
| Multi-tenancy | RLS enforced via non-superuser role; per-request `app.current_user_id` | alice can't read bob even with explicit WHERE |
| Trust gate | agents can only write `status='pending'` instructions | enforced at API; pipeline test asserts it |
| Tool/procedural memory | tool calls captured to episodic; procedural agent distills tool lessons (opt-in via `coding` preset) | verified: tool failure → procedure memory |
| Memory-bulk control | dedup→reinforce at write; one extract per completed turn; current-exchange-only logging; tool-result truncation | clean single-exchange capture verified |
| Quality | 96% (23/24) on the multi-session eval, tied with Mem0 | `bench/quality` |

## 6. What's deliberately NOT done (known gaps)

Be honest with users about these:

- **No guardrails / safety layer.** No PII detection/redaction, no prompt-injection
  defense on injected memory, no content filtering. (Injected memories are currently
  trusted text.)
- **No observability.** No tracing/metrics/dashboards; debugging is via logs.
- **Temporal/contradiction handling.** A changed fact (e.g. a promotion, a move)
  is not *superseded* — both old and new linger; the eval shows this costs recall
  on `update` questions. Needs contradiction detection + memory versioning.
- **Scale unproven.** All numbers are single-client/local. No load/throughput/chaos
  testing at scale; no PgBouncer/read-replica setup yet.
- **Dual-write gap.** `/v1/ingest` writes the episodic row and enqueues separately
  (not atomic). An outbox pattern would close it.
- **Ecosystem.** Python SDK only (no JS/TS); no streaming post-hook handling; no
  framework adapters. (Mem0 now leads here — async-by-default, TS SDK, ~21
  integrations.) Procedural/tool memory works but only for OpenAI-style tool loops
  that round-trip through the wrapped client.
- **Faithfulness cost.** The gate doubles extraction LLM calls (extract + verify).

## 7. Extending it

- **Add a memory backend to the eval** (e.g. Letta, Zep): implement the adapter
  surface in `bench/quality/adapters.py` (`setup` / `ingest` / `wait_ready` /
  `retrieve`) and register it in `ADAPTERS`. Note these are agent/service systems
  and need their own running server + an isolated Python env.
- **Add a DB migration:** drop `NNN_name.sql` in `memgram/memory/migrations/`;
  the runner applies new files in order and records them. `{EMBED_DIMS}` is
  substituted from `MEMGRAM_EMBED_DIMS`.
- **Swap the LLM / embedder:** point `MEMGRAM_LLM_BASE_URL` at any OpenAI-compatible
  server; set `MEMGRAM_EMBED_MODEL` / `MEMGRAM_EMBED_DIMS` (the latter *before*
  first migrate, since the vector column dimension is fixed at creation).
- **Tune behavior:** presets + overrides resolve in `memgram/presets.py`; the worker
  reads the same preset via `MEMGRAM_PRESET`.

## 8. Repo map

```
memgram/
  sdk/         Memgram, proxy intercept, context assembler, config, HTTP client
  api/         FastAPI app + routes (ingest, instructions, memory, proposals)
  agents/      base + extractor(+faithfulness), reflection, proposer, decay, summarizer
  memory/      store (dedup/reinforce), retriever (ranked), embedder, migrations/
  worker/      queue (Streams), dispatcher (K consumers + reclaim + DLQ), scheduler
  presets.py   defaults → preset → overrides merge
dashboard/     Next.js UI
bench/         bench.py (latency), compare_mem0.py (latency vs Mem0), quality/ (recall eval)
examples/      demo_persistence.py, qa_rag_app.py
tests/         test_logic.py, test_thin_slice.py, test_pipeline.py
docker/        API image + opt-in Postgres+pgvector+AGE image
```

## 9. Environment variables (the ones that matter)

| Var | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | LLM + embeddings | — |
| `MEMGRAM_API_KEY` | bearer token the SDK uses to call the API | `mgram_dev_key` |
| `DATABASE_URL` / `REDIS_URL` | Postgres / Valkey connection | compose-wired |
| `MEMGRAM_DB_PORT` / `MEMGRAM_QUEUE_PORT` | host port remap if 5432/6379 are taken | 5432 / 6379 |
| `MEMGRAM_EMBED_DIMS` / `MEMGRAM_EMBED_MODEL` | embedding dims/model (dims set before first migrate) | 1536 / text-embedding-3-small |
| `MEMGRAM_FAST_MODEL` / `MEMGRAM_QUALITY_MODEL` | extractor/summarizer vs reflection/proposer | gpt-4o-mini / gpt-4o |
| `MEMGRAM_LLM_BASE_URL` | OpenAI-compatible endpoint (local models) | OpenAI default |
| `MEMGRAM_PRESET` | minimal\|chatbot\|coding\|enterprise\|privacy\|custom | (defaults) |
| `MEMGRAM_WORKER_CONCURRENCY` | parallel job consumers per worker | 8 |
| `MEMGRAM_FAITHFULNESS` | `0` disables the anti-hallucination gate | on |
| `MEMGRAM_FAKE_LLM` / `MEMGRAM_FAKE_REDIS` | zero-infra dev (canned LLM / in-proc queue) | off |
