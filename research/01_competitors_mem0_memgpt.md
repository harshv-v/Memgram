# Competitor Deep-Dive: Mem0 & MemGPT (from the papers)

Grounded in the downloaded papers (`research/papers/`). The headline: **both solve
the contradiction/update problem with an explicit memory-mutation step that Memgram
lacks.** Memgram only ever ADDs/reinforces; it never UPDATEs or DELETEs.

---

## Mem0 (arXiv 2504.19413) — the system we benchmark against

### Two-phase pipeline: Extraction → **Update**
1. **Extraction.** Prompt = `(global conversation summary S, last m messages, new
   message pair)`. An LLM ϕ extracts candidate facts Ω = {ω₁…ωₙ}. (An *async*
   summary module keeps S fresh without blocking — same idea as our summarizer.)
2. **Update phase (THE KEY PART we're missing).** For *each* candidate fact ωᵢ:
   - retrieve the **top-s = 10** semantically similar existing memories,
   - present `(ωᵢ, those 10 memories)` to an LLM via a **function-calling "tool
     call"**, which picks **one of four operations**:
     - **ADD** — no semantically equivalent memory exists → create.
     - **UPDATE** — augment an existing memory with complementary info.
     - **DELETE** — remove a memory **contradicted** by the new fact.
     - **NOOP** — no change needed.
   - No separate classifier — the LLM's reasoning selects the op from the
     candidate↔existing relationship.

### Mem0g (graph variant)
- Memories as a directed labeled graph: entities (nodes, typed + embedded +
  timestamped) and relationships (edges like `lives_in`, `prefers`).
- **Conflict detection + an LLM "update resolver"** that marks obsolete
  relationships **INVALID rather than deleting them** — soft-invalidate to preserve
  *temporal reasoning* (you can still ask "where did they live before?").

### Config / eval (useful specifics)
- m = 10 recent messages, s = 10 similar memories, **gpt-4o-mini**, temperature 0.
- Benchmarked on **LOCOMO**: 10 conversations × ~600 dialogues × ~26k tokens,
  ~200 Q each. Categories: **single-hop, multi-hop, temporal, open-domain**.
- Metric: **LLM-as-a-Judge**, **10 runs ± std** (they explicitly average 10 runs
  because J is stochastic — this is why our 24-Q eval swings ±4% run-to-run).
- Claims vs full-context: ~26% higher J, **91% lower p95 latency**, >90% token saving.

### Why this matters for us — DIRECT diagnosis of our regression
My contradiction attempt asked: *"which existing memories are obsolete, by which
new fact?"* — an open, error-prone framing that over-archived (80%→20%). Mem0's
framing is **per-candidate operation selection**: for each NEW fact, look at its
own top-s neighbours and choose ADD/UPDATE/DELETE/NOOP. The decision is local,
bounded, and structured. That's the fix: **adopt the operation-selection framing**,
not "pick rows to kill." And soft-invalidate (we already have `superseded_by`).

---

## MemGPT / Letta (arXiv 2310.08560) — LLM as an Operating System

### Hierarchical memory (the "OS" analogy)
- **Main context** (in the window): system instructions (read-only) + **working
  context** (read-write, small editable facts — persona/human) + **FIFO queue** of
  recent messages.
- **External context** (out of window): **recall storage** (full conversation
  history, searchable) + **archival storage** (arbitrary-length vector DB).

### Self-directed memory editing via function calls (THE KEY IDEA)
- The LLM **edits and searches its own memory** through functions:
  `core_memory_append`, `core_memory_replace`, `archival_memory_insert/search`,
  `conversation_search`. **Memory edits are self-directed, in the hot loop.**
- **Contradiction handling = `core_memory_replace`**: the agent overwrites a stale
  working-context fact ("lives in Berlin") with the new one ("lives in Munich") as
  its "evolving understanding" changes — *in-place edit during the conversation*.
- **Memory pressure / paging**: at ~70% context → "memory pressure" warning so the
  LLM saves important items out; at 100% → flush + **recursive summary** of evicted
  messages. (This is the *synchronous* summarizer behavior we don't have.)
- **Heartbeats** (`request_heartbeat=true`): chains function calls for **multi-step /
  agentic retrieval** — search, read, search again — to answer multi-hop queries.

### Contrast in *where* memory is managed
- **MemGPT**: synchronous, agent-driven, in the conversation loop (the model spends
  tokens deciding to edit memory). Powerful, but adds latency + token cost per turn
  and requires the agent to follow a complex function protocol.
- **Mem0 / Memgram**: background, pipeline-driven (memory managed off the hot path).
  Cheaper + faster per turn, but resolution lags the conversation.

---

## The one-line takeaways

| Capability | MemGPT | Mem0 | **Memgram (today)** |
|---|---|---|---|
| Add memory | ✅ archival_insert | ✅ ADD | ✅ extractor |
| **Update/merge memory** | ✅ core_memory_replace | ✅ **UPDATE** | ❌ |
| **Delete/invalidate stale** | ✅ replace overwrites | ✅ **DELETE** (soft in graph) | ⚠️ infra only (`superseded_by`), heuristic disabled |
| Dedup repeats | implicit | implicit | ✅ dedup→reinforce |
| Reinforcement/decay | ❌ | ❌ | ✅ (Ebbinghaus) — *our* differentiator |
| Multi-hop / graph | ✅ heartbeats | ✅ Mem0g graph | ❌ |
| Synchronous summarize at pressure | ✅ | (async) | ❌ (async) |
| Importance/salience at write | ❌ | partial | ❌ |

**Where Memgram is unique:** reinforcement + Ebbinghaus decay (active forgetting) —
neither competitor models memory *strength over time*. **Where Memgram is behind:**
the UPDATE/DELETE operation (contradiction), graph/multi-hop, and pressure-time
summarization.
