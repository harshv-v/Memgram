# Memgram

**A zero-config cognitive memory layer for AI agents. Wrap your LLM call — memory works instantly.**

Every agent session starts from zero. Memgram is the missing layer: it remembers what
the user told you, decides what's worth keeping, lets old facts fade, and hardens
repeated patterns into permanent habits — all behind two lines of code, with any LLM client.

```python
import openai
from memgram import Memgram

mem    = Memgram(api_key="mgram_dev_key", agent_name="my-agent")   # line 1
client = mem.wrap(openai.OpenAI())                                # line 2

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Help me sort a list."}],
    user_id=current_user.id,          # the only new parameter
)
# streaming, tools, function-calling, response parsing — all unchanged
```

Tell it your preferences in one session. Close it. Open a fresh session. It already
knows — concise answers, your language, your stack — without you re-telling it.

---

## Why it's different

- **No framework dependency.** Wraps OpenAI, Anthropic, Ollama, vLLM, Groq — anything
  OpenAI-compatible. No LangChain, no LangGraph, no LlamaIndex. The agent base class is
  ~60 lines; every background agent is a prompt, a JSON response, and a retry loop.
- **Memory is an active process.** Memories decay when unused (Ebbinghaus forgetting
  curve), strengthen when reinforced, and consolidate into habit when patterns repeat —
  instead of accumulating forever and polluting retrieval.
- **Instruction memory is first-class.** User-editable, priority-ordered rules injected
  first into every prompt. Agents can *propose* new ones, but **only a user can approve
  them** — the trust gate against memory poisoning.
- **One Postgres for everything.** pgvector for similarity, Apache AGE for the entity
  graph (opt-in), plain SQL for the rest. One connection string, row-level security once.

## The four memory pillars

| Pillar | What | Decay | Backed by |
|---|---|---|---|
| **Instruction** | User rules, always injected first | never | Postgres (+ Redis cache) |
| **Semantic** | Facts, entities, preferences | medium (days) | Postgres + pgvector |
| **Episodic** | Timestamped interaction logs | fast (hours) | Postgres (time-indexed) |
| **Procedural** | Tool success/failure patterns | slow (weeks) | Postgres + pgvector *(opt-in)* |

Retention follows `R = exp(-t / S)` — stability `S` grows with each reinforcement;
corrections carry `emotional_weight = 2.0` so mistakes persist longer, like humans.
Cross the habit threshold (default 7 reinforcements) and the agent proposes a permanent
instruction for you to approve.

## The six background agents

Run in a worker, never on the hot path (the summarizer can run inline when context is
over budget). Each is plain Python.

| Agent | Trigger | Does |
|---|---|---|
| **Extractor** | every interaction | facts / preferences / entities / corrections → semantic memory |
| **Summarizer** | context over threshold | compress a long conversation → one structured session row (~85% fewer tokens) |
| **Reflection** | every N interactions / 24h | raw logs → durable insights; flags habit candidates |
| **Proposer** | after reflection | habit candidate → **pending** instruction proposal (+ webhook) |
| **Decay** | nightly 02:00 UTC | recompute retention, re-tier, flag stale instructions — pure SQL, $0 |
| **Router** | retrieval time | scope filter on every SELECT — pure rules |

---

## Architecture

```
your code ──► mem.wrap(client).chat.completions.create(...)
                 │
   ┌─────────────┴───────────────── HOT PATH (sync, <5ms target) ──────────────┐
   │ 1. assembler enriches messages:  instructions  →  semantic memories        │
   │ 2. the real LLM call fires, unchanged                                      │
   │ 3. the exact same response object is returned                              │
   │ 4. post-hook fires fire-and-forget ──► POST /v1/ingest  (never blocks)     │
   └────────────────────────────────────────────────────────────────────────────┘
                                          │
   ┌────── BACKGROUND (Redis/Valkey queue → worker, zero impact on hot path) ───┐
   │ ingest → episodic log + `extract` job                                      │
   │ extract → reflect (cadence) → propose → (nightly) decay                    │
   └────────────────────────────────────────────────────────────────────────────┘
```

```
memgram/
├── sdk/         Memgram, proxy intercept, context assembler, config, HTTP client
├── agents/      base + extractor, summarizer, reflection, proposer, decay
├── memory/      store (dedup→reinforce), retriever (ranked), embedder, migrations
├── worker/      Redis/Valkey queue (no Celery), dispatcher, scheduler
└── api/         FastAPI: ingest, instructions, memory (+GDPR export/forget), proposals
dashboard/       Next.js UI: view/edit instructions, browse memory, approve proposals
docker/          API image + opt-in Postgres+pgvector+AGE image
```

**Stack, deliberately boring:** FastAPI + asyncpg (no ORM), Redis/Valkey queue (~80 lines,
no Celery), Postgres + pgvector (HNSW index), structured `json_object` outputs everywhere.
100% open-source infrastructure; OpenAI is the default brain with a documented escape hatch
to local models (`MEMGRAM_LLM_BASE_URL`, `MEMGRAM_EMBED_MODEL`, `MEMGRAM_EMBED_DIMS`).

---

## Quickstart (Docker — the whole stack)

```bash
cp .env.example .env          # set OPENAI_API_KEY (or MEMGRAM_FAKE_LLM=1 for zero-cost dev)
docker compose up             # db (pgvector) + queue (Valkey) + migrate + api + worker
```

API on `http://localhost:8000` (`/health`, `/docs`). Then run the demo:

```bash
pip install -e .
OPENAI_API_KEY=sk-...  python examples/demo_persistence.py
```

### The dashboard

```bash
cd dashboard && npm install && npm run dev     # http://localhost:3000
```

View and edit instructions, browse semantic memories (with decay tier + retention),
and approve or dismiss agent proposals.

### Local dev with no infrastructure at all

`MEMGRAM_FAKE_LLM=1` (canned agent + embedding responses) and `MEMGRAM_FAKE_REDIS=1`
(in-process fakeredis) let the full pipeline run with zero external calls — used by the tests.

### Opt-in: the Apache AGE graph layer

The default `db` service is `pgvector/pgvector:pg16` (vectors + relational — everything
v1 needs). To enable the entity graph, switch `db` to `docker/postgres-age.Dockerfile`,
which builds one Postgres with pgvector **and** Apache AGE.

---

## Tests

```bash
# Logic suite — pure, no infra:
MEMGRAM_FAKE_LLM=1 python tests/test_logic.py

# Thin slice — needs the API up against Postgres:
MEMGRAM_API_KEY=mgram_dev_key python tests/test_thin_slice.py

# Background pipeline (extract→retrieve→dedup→reflect→propose→decay→summarize):
DATABASE_URL=postgresql://memgram:memgram@localhost:5432/memgram \
  MEMGRAM_FAKE_LLM=1 MEMGRAM_FAKE_REDIS=1 python tests/test_pipeline.py
```

## For reviewers / testers

Thanks for trying Memgram. The fastest way to see what it does:

```bash
git clone https://github.com/harshv-v/Memgram.git && cd Memgram
cp .env.example .env          # paste your OPENAI_API_KEY (or set MEMGRAM_FAKE_LLM=1)
docker compose up --build     # db + queue + migrate + api + worker

pip install -e .
OPENAI_API_KEY=sk-...  python examples/qa_rag_app.py   # memory forms, recalls, isolates
cd dashboard && npm install && npm run dev             # http://localhost:3000
```

`examples/demo_persistence.py` is the 30-second version: tell it your stack in
one process, restart, and it already knows in the next.

**What I'd love feedback on (open an Issue):**

- **Extraction quality** — are the facts/preferences it pulls from your real
  conversations the *right* ones? Too aggressive, too sparse, wrong?
- **The habit threshold** — a preference is proposed as a permanent instruction
  after 7 reinforcements. Does 7 feel right for your usage?
- **Integration friction** — was the two-line wrap actually drop-in for your app?
- **Anything that broke** — paste the error; the worker/API logs help a lot.

Issues and PRs welcome. If something doesn't run, that's the most useful bug of all.

For a full engineering walkthrough (architecture, what's hardened, known gaps, how
to extend) see **[HANDOVER.md](HANDOVER.md)**; for the *why* behind each design
decision see **[DESIGN.md](DESIGN.md)**; for measured numbers see
**[BENCHMARKS.md](BENCHMARKS.md)**.

## Security & trust

Agents can **never** write an active instruction — only `status='pending'` proposals;
enforced at the API layer (there is no endpoint that lets an agent self-promote).
Postgres row-level security scopes every table to `(project_id, user_id)`. GDPR endpoints:
`GET /v1/memory/export/{user_id}` and `DELETE /v1/memory/user/{user_id}` (hard delete).

## Configuration

Behaviour is set with a **preset** plus overrides (three-layer merge: platform
defaults → preset → your overrides):

```python
mem = Memgram(api_key="mem_...", agent_name="coder",
               preset="coding",                 # minimal|chatbot|coding|enterprise|privacy|custom
               memory_budget=4000,               # any override
               decay={"habit_threshold": 10})
```

`minimal` is instruction-store-only (zero background cost); `coding` turns on all
four pillars. The worker honors the same preset via `MEMGRAM_PRESET`, so feature
flags match end to end. Env knobs: `DATABASE_URL`, `REDIS_URL`, `MEMGRAM_API_KEY`,
`OPENAI_API_KEY`, the local-LLM escape hatch (`MEMGRAM_LLM_BASE_URL` /
`MEMGRAM_EMBED_MODEL` / `MEMGRAM_EMBED_DIMS`), model tiers (`MEMGRAM_FAST_MODEL` /
`MEMGRAM_QUALITY_MODEL`), summarizer trigger (`MEMGRAM_SUMMARIZE_THRESHOLD`),
`MEMGRAM_WEBHOOK_ON_PROPOSAL`, and dev switches `MEMGRAM_FAKE_LLM` /
`MEMGRAM_FAKE_REDIS`. See `.env.example`.

Multi-tenancy is enforced by per-request row-level security (migration 004 sets
`app.current_user_id`); the background worker runs unscoped for cross-user sweeps.

## Status & roadmap

v1 is self-hosted, single-agent, four-pillar core (procedural opt-in). Deliberately **not**
built yet: Apache AGE graph queries, TimescaleDB episodic archival, the multi-agent router,
streaming support, enterprise RBAC/SSO, and the managed cloud. Those are driven by real usage —
get ~10 developers on it first.

Inspired by the article *"Your AI Agent Has Amnesia: The Blueprint for Cognitive Memory
Architectures."*

## License

MIT © 2026 Harsha Vanukuri
