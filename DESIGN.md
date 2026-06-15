# Memgram — Design Decisions (what / how / why)

This explains *why* Memgram is built the way it is. For how to run it see
[HANDOVER](HANDOVER.md); for numbers see [BENCHMARKS](BENCHMARKS.md). Each section:
**what** we do, **how**, **why**.

---

## 1. Two-line integration via client proxy
- **What:** `mem.wrap(openai_client)` returns a drop-in client; you only add `user_id`.
- **How:** the proxy intercepts `chat.completions.create`, everything else passes
  through untouched ([sdk/proxy.py](memgram/sdk/proxy.py)).
- **Why:** memory should require zero rewrite. If adopting it is a migration,
  nobody adopts it.

## 2. Hot path vs background split
- **What:** retrieval+injection happen inline; extraction/reflection/decay happen
  in a worker.
- **How:** after the response, a fire-and-forget post-hook posts to `/v1/ingest`,
  which enqueues a job and returns `202`. The user never waits on memory writes.
- **Why:** an LLM call is already slow; memory must never add user-visible latency.
  Measured hot-path overhead is ~2 ms (instructions) + the embedding round-trip.

## 3. Four pillars, different lifespans
- **What:** instructions (never decay), semantic (days), episodic (hours),
  procedural (weeks).
- **How:** separate tables / `memory_type`; decay is a nightly SQL pass.
- **Why:** not all memory is equal. A standing rule must outlive a passing fact;
  raw logs are noise within hours. Modeled on human memory.

## 4. Active memory: collapse at write, fade over time (NOT "store then prune")
- **What:** repeated/near-duplicate memories don't accumulate — they reinforce a
  single row; unused memories decay and archive.
- **How:** every semantic/procedural write goes through **dedup→reinforce**
  ([store.py](memgram/memory/store.py)): if a new memory is within cosine distance
  0.15 of an existing one, increment its `reinforcement_count` + reset decay
  instead of inserting. Retrieval is **top-k**, so prompt size is bounded by k,
  not by how many times something was said. A nightly Ebbinghaus pass
  (`R = exp(-t/stability)`) re-tiers active→fading→archived.
- **Why:** this is the core thesis. "Store everything and sort it later" is what
  bloats context and pollutes retrieval. Bounding bulk **at write time** is
  cheaper and keeps recall sharp. Verified: 12 identical writes → 1 row + 11
  reinforcements.

## 5. Trust gate — agents propose, humans approve
- **What:** background agents can only create `status='pending'` instructions; a
  human promotes them to `active`.
- **How:** enforced at the API layer — no endpoint lets an agent write an active
  instruction.
- **Why:** instruction memory steers every future prompt. Letting an agent
  self-promote is a memory-poisoning / prompt-injection hole. The human gate is
  the backstop.

## 6. Faithfulness gate on extraction
- **What:** extracted candidate memories are checked against the transcript; any
  not supported are dropped.
- **How:** a second strict LLM "fact-checker" pass ([agents/extractor.py](memgram/agents/extractor.py));
  each stored memory keeps a provenance excerpt.
- **Why:** a *false* memory is worse than no memory — it silently poisons every
  later prompt. Costs one extra LLM call per extraction; worth it. Toggle with
  `MEMGRAM_FAITHFULNESS=0`.

## 7. Tool / procedural memory and its bulk strategy
- **What:** tool calls and results are captured; the procedural agent distills
  reusable lessons ("calling X without Y times out") into procedure memory.
- **How:** the SDK flattens tool turns into `tool_call` / `tool_result` episodic
  rows; a tool-bearing completed turn triggers the procedural agent, which writes
  `memory_type='procedure'` (through the same dedup→reinforce path).
- **Why & how we keep it from bulking up** (a tool fires the same way constantly):
  - **Dedup→reinforce** (see §4): the same tool lesson collapses into one
    reinforced row, not thousands.
  - **Extract once per completed turn:** a multi-call tool loop has no
    `response_text` on intermediate round-trips, so extraction runs once on the
    final turn — not once per tool call (saves redundant LLM work).
  - **Log only the current exchange:** ingest logs from the last user message
    onward, so resending full history each turn doesn't duplicate episodic.
  - **Truncate tool results:** tool outputs are capped before storage — we keep
    the success/failure shape, not a 10k-token payload.
  - **Episodic is fast-decay and never injected** into prompts (only semantic +
    instructions are), so raw tool volume is storage, not prompt tokens.

## 8. Durable queue on Valkey Streams
- **What:** at-least-once job delivery; a crashed worker's job is reclaimed; a
  job that keeps failing goes to a dead-letter stream.
- **How:** Streams + consumer groups; ack-on-success, `XAUTOCLAIM` for reclaim
  ([worker/queue.py](memgram/worker/queue.py)).
- **Why:** the first version used a plain list pop — a crash mid-job lost the job
  silently. Memory you might-or-might-not have stored is not memory.

## 9. Concurrency with correctness
- **What:** the worker runs K consumers; dedup stays correct under concurrency.
- **How:** consumer-group members process in parallel; `upsert_semantic` runs its
  check-then-insert under a per-(project,user,agent) advisory lock.
- **Why:** serial extraction was the latency bottleneck (~18 s → ~4.7 s for 8
  facts). But concurrency creates a check-then-insert race that would duplicate
  memories — the lock closes it (12 concurrent identical → 1 row).

## 10. One Postgres + Valkey, no frameworks
- **What:** pgvector for similarity, plain SQL for the rest, an ~80-line queue, no
  ORM / Celery / LangChain.
- **How:** asyncpg + FastAPI; agents are a prompt + JSON + retry loop.
- **Why:** boring infrastructure is operable and auditable. Every dependency you
  add is a thing that breaks at 3am. The differentiator is the memory model, not
  the plumbing.

## 11. Multi-tenancy via row-level security
- **What:** every table is scoped to its tenant; the API can't read across users
  even if a query forgets a `WHERE`.
- **How:** Postgres RLS; the runtime connects as a **non-superuser** role and sets
  `app.current_user_id` per request; the worker runs unscoped for cross-user sweeps.
- **Why:** defense in depth. WHERE-clause-only isolation is one typo from a leak.
