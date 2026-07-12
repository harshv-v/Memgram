# 03 — Context-engineering landscape: OpenClaw, Claude, compression (June 2026)

How the systems people actually use manage, truncate, and compress memory — and an
integrate-or-not verdict for each idea. Companion to 01 (competitors) and 02 (gaps).

---

## 1. OpenClaw (ex-Moltbot/Clawdbot)

**Model: files + compaction + a "flush before you forget" reflex.**

- Memory is plain Markdown on disk: daily notes (`memory/YYYY-MM-DD.md`) and a
  long-term `MEMORY.md` with structured headings (Preferences / Projects / Decisions).
- When a session nears the context limit it **compacts** — older messages get
  summarized/dropped. Anything not yet written to disk is destroyed.
- The clever bit: a **soft-threshold memory flush**. When token estimate crosses
  `contextWindow − reserveTokensFloor − softThreshold`, OpenClaw injects a silent
  agentic turn telling the model "write down what matters NOW, before compaction."
- Limits: 20k chars per bootstrap file, 150k aggregate; over that → truncated.
- Weakness (Mem0's own critique): preservation depends on the compaction *event*
  firing and the model choosing well under pressure. Turn-level capture is safer.

**What we take:** the flush-before-truncation *reflex* is genuinely good design —
it's a warning signal, not a storage model. **What we don't:** files-as-memory.
No decay, no dedup, no reinforcement, no multi-user isolation, unbounded growth
until truncation. Memgram already captures at turn level (extract on every
interaction) — architecturally the stronger position.

## 2. Claude (Anthropic API + Claude Code)

**Model: client-side memory files + context editing + compaction, orchestrated.**

- **Memory tool** (beta): Claude gets create/read/update/delete over a memory
  directory *you* store. File-based, developer-owned backend.
- **Context editing**: when context nears a threshold, stale tool results get
  cleared; Claude receives an automatic warning first so it can save what matters
  to memory files — the same flush reflex as OpenClaw, productized.
- **Compaction**: server-side summarization of older turns is the recommended path.
- Anthropic's numbers: memory + context editing = **+39%** on agentic search,
  **84% token reduction** on a 100-turn eval. Claude Code additionally uses
  CLAUDE.md (bootstrap instructions — the same concept as our instruction pillar).

**What we take:** validation. "Instructions injected first" (CLAUDE.md), "save
before truncate", "summarize old turns" — Memgram ships all three equivalents.
**What's notable:** Anthropic made the *agent* responsible for memory hygiene via
tools; Memgram makes the *infrastructure* responsible. Both are defensible; ours
needs zero model cooperation, theirs needs zero infra. Not something to copy —
something to position against.

## 3. Fast compression — CompLLM, LLMLingua, KV-cache quant

| Technique | What it is | Speed/win | Where it lives |
|---|---|---|---|
| **CompLLM** (arXiv 2509.19228) | *Soft* compression: segments of 20 token-embeddings → 10 "concept embeddings" via a LoRA on the LLM itself. Linear over context, cacheable per segment. | 2× compression → up to 4× faster TTFT, −50% KV cache | **Inside the serving stack** — needs access to the model's embedding layer |
| **LLMLingua / LLMLingua-2** (Microsoft) | *Hard* compression: a small classifier drops low-value tokens from prompt text. Task-agnostic, model-agnostic, output is still text. | 2–5× compression, ~2.9× lower end-to-end latency; prod deployments report 4–10× | **Between retrieval and the LLM call** — pure text-in/text-out |
| **TurboQuant / KV-quant** | Quantize the KV cache (PolarQuant+QJL etc.), ~6× KV memory, ~8× faster attention | training-free | **Inference server only** (vLLM etc.) |

**Verdicts:**
- **CompLLM: no.** It operates on embeddings inside the model — impossible from
  our seat (we wrap the client; OpenAI won't accept concept embeddings). Only
  relevant if a self-hoster runs their own vLLM — and then it's *their* layer, not ours.
- **KV-cache quant: no** (same reason — the design doc already parked TurboQuant).
- **LLMLingua-2: yes, later, as an optional squeeze stage.** It's the only one
  that fits our architecture: text in, text out, sits exactly where our assembler
  builds the memory block. Concretely: when the assembled memory block exceeds
  `memory_budget`, run it through LLMLingua-2 instead of dropping items. Optional
  dependency, off by default, `features={"compress": true}`. Note we're *already*
  the cheap path — injecting 5 ranked memories (~a few hundred tokens) instead of
  26k of history IS the compression story; LLMLingua polishes the margin, it
  doesn't create it. Priority: after launch feedback, not before.

## 4. Do we integrate the frameworks themselves?

No integration *dependency* — that's the founding rule (no LangChain/LangGraph
imports). But two cheap ecosystem moves matter:
1. **Adapters/recipes, not dependencies** — a docs page + tiny example showing
   Memgram under LangGraph, CrewAI, and the OpenAI Agents SDK (they all take an
   OpenAI-compatible client, so `mem.wrap()` already works; prove it with examples).
2. **An OpenClaw plugin** is a real distribution channel (Mem0 built one within
   weeks of OpenClaw going viral). Small surface: turn-level capture → our ingest.

## 5. Competitive readiness (honest)

Where we stand vs the four incumbents (Mem0 48k★/$24M raised, 21 integrations,
new 2026 single-pass algorithm; Zep temporal graph; Letta agent-OS; LangMem):

**Genuinely competitive:** recall quality (96%, tied with Mem0 on our eval);
hot-path latency (local ONNX embedder kills the network hop); decay/reinforcement/
habit formation (still nobody ships this); trust-gated instruction memory (unique);
self-host story (one Postgres, RLS, GDPR endpoints, no cloud dependency).

**Not competitive yet:** ecosystem (Python-only vs Mem0's 21 integrations + TS SDK);
temporal reasoning (Zep's whole moat; our contradiction-v2 is built but off by
default pending eval win); observability (logs only); scale proof (single-client
numbers only); community (0 stars — unlaunched).

**Verdict: ready for early adopters, not ready to out-feature Mem0 — and that's
fine.** We don't win a feature war in v0.1; we win a *positioning* war: the
cognitive memory layer (decay + habits + trust gate) that self-hosts on one
Postgres. Ship, get the first 10 users, and let their pull decide whether TS SDK,
OpenClaw plugin, or temporal reasoning comes first.

### Recommended order (post-launch)
1. Launch as-is (blog + HN + r/LocalLLaMA; the article audience is the wedge).
2. OpenClaw plugin (distribution, small surface).
3. Framework recipe pages (zero code, kills the "does it work with X?" objection).
4. Contradiction-v2 eval push — beat the update axis, turn it on by default
   (answers Zep's temporal story at the fact level).
5. LLMLingua-2 squeeze stage (optional compress).
6. TS SDK only when users ask with volume.
