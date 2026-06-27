# Memgram Research

Research-driven analysis to guide Memgram's next phase — grounded in the papers in
`papers/` and the leading systems (Mem0, MemGPT/Letta), not vibes.

## Read in order
1. **[00_memgram_deep_dive.md](00_memgram_deep_dive.md)** — what Memgram actually
   does today (agents, memory model, verified numbers, the honest gaps).
2. **[01_competitors_mem0_memgpt.md](01_competitors_mem0_memgpt.md)** — how Mem0 and
   MemGPT work, from their papers; the capability matrix.
3. **[02_gap_analysis_and_roadmap.md](02_gap_analysis_and_roadmap.md)** — ranked gaps
   and the research-grounded plan, centered on rebuilding contradiction resolution.
4. **[papers/README.md](papers/README.md)** — the downloaded paper corpus + why each.

## The one finding that drove this
Naively asking an LLM "which memories are now obsolete?" **regressed** our temporal
`update` score from 80% → 20%. Mem0's proven approach is fundamentally different:
per new fact, choose one bounded operation — **ADD / UPDATE / DELETE / NOOP** —
against its nearest neighbours. That reframing is the fix. Memgram already has the
soft-invalidation infra (`superseded_by`); only the decision algorithm was wrong.

## What stays Memgram's own
Reinforcement + Ebbinghaus **decay** (active forgetting) — neither Mem0 nor MemGPT
models memory strength over time. Keep the moat; close the UPDATE/DELETE + benchmark
gaps.

> Not in scope / we don't have: Google TurboQuant (inference-layer KV-cache
> compression — a model-serving optimization, not an app-layer memory technique).
