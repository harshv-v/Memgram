# Contributing to Memgram

Thanks for trying Memgram. It's early and real-world usage is exactly what drives the
roadmap — bug reports, rough edges, and "this didn't do what I expected" are all valuable.

## Getting set up

```bash
git clone <your-fork>
cd memgram
python -m venv .venv && source .venv/bin/activate
pip install -e ".[backend,dev]"
cp .env.example .env
docker compose up        # Postgres(pgvector) + Valkey + api + worker
```

For zero-infra hacking, set `MEMGRAM_FAKE_LLM=1` and `MEMGRAM_FAKE_REDIS=1` — the whole
pipeline runs with no external calls.

## Running tests

```bash
MEMGRAM_FAKE_LLM=1 python tests/test_logic.py                         # pure logic, no infra
MEMGRAM_API_KEY=mgram_dev_key python tests/test_thin_slice.py           # needs the API + Postgres
DATABASE_URL=... MEMGRAM_FAKE_LLM=1 MEMGRAM_FAKE_REDIS=1 \
  python tests/test_pipeline.py                                      # background agents, real Postgres
```

## Design principles (please keep these)

- **No agent frameworks.** No LangChain / LangGraph / LlamaIndex / Celery / ORM. An agent
  is a prompt, a `response_format: json_object` call, and a retry loop.
- **The hot path is sacred.** Nothing added to `proxy`/`assembler` may block the LLM call;
  all intelligence is background work behind the queue.
- **The trust gate is non-negotiable.** Agents may only write `status='pending'` proposals.
  There must be no code path that lets an agent write an active instruction.
- **One Postgres.** Reach for pgvector / AGE / a column before reaching for a new datastore.

## What's most useful right now

Real integrations against real LLMs: extraction quality, reflection insight quality, and
calibration of the habit threshold (default 7) — these can only be tuned with real data.
Open an issue with what you saw.

## Where things stand

See the roadmap in the README. The "do not build yet" list is intentional — please open an
issue to discuss before adding anything on it.
