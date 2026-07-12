# Six Months Ago I Said Your AI Agent Has Amnesia. Here's the Cure — Open Source, Benchmarked, on One Postgres.

*I published the blueprint in January. Today the system exists: memory that decays when unused, strengthens when reinforced, and hardens into habits — behind two lines of code. I benchmarked it against Mem0 and I'm publishing the harness, including where I lose.*

---

In January I wrote [Your AI Agent Has Amnesia: The Blueprint for Cognitive Memory Architectures](LINK-TO-ORIGINAL-ARTICLE). The thesis was simple: the reason your agent forgets everything between sessions is not a model problem — GPT, Claude, and Gemini are all capable enough. It's an infrastructure problem. There is no layer that knows what to remember, how long to keep it, when to forget it, and how to turn repeated patterns into permanent behaviour.

That article was a design document. People asked the obvious question: *where's the code?*

Fair. Today: **Memgram** — open source, MIT, self-hosted, tested against Mem0 on a reproducible benchmark I'm publishing alongside the code. This is the story of what I built, what the blueprint got right, what it got wrong, and where it honestly stands against the funded competition.

## The problem, thirty seconds

Every agent session starts from zero. Your user says "keep it short, I write Rust" on Monday; on Tuesday the agent explains recursion in verbose Python. Developers compensate by stuffing 26,000 tokens of conversation history into every API call. Users compensate by re-explaining themselves, forever.

Four companies dominate the fix — Mem0, Zep, Letta, LangMem — and they all share one architectural assumption I think is wrong: **memory as append-only storage.** Facts accumulate forever. Your old city, your old job, your abandoned framework — all retrieved alongside current facts, silently degrading every prompt. Human memory doesn't work that way, and there are good computational reasons why.

## Two lines

The founding constraint was that integration must be invisible:

```python
import openai
from memgram import Memgram

mem    = Memgram(api_key="...", agent_name="my-agent")   # line 1
client = mem.wrap(openai.OpenAI())                        # line 2

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Help me sort a list."}],
    user_id=current_user.id,        # the only new parameter
)
```

Streaming, tool calls, response parsing — unchanged. The proxy intercepts one method, injects the user's memory before the call, and mines the exchange in the background after it. Wrap Anthropic's native client, Ollama, vLLM, Groq, Gemini — same two lines.

Tell it your preferences in one session. Kill the process. Start a new one. It already knows. That demo — a real process restart, not a cleared variable — is `examples/demo_persistence.py` in the repo, and it runs in 30 seconds.

## Memory as a process, not a warehouse

This is the part nobody else ships, and the reason I built this instead of adopting Mem0.

**Memories decay.** Every memory carries a retention score following the Ebbinghaus forgetting curve — `R = exp(-t/S)` — where stability `S` grows each time a memory is accessed. Episodic details fade in hours. Semantic facts fade in days if never reinforced. A nightly job (pure SQL, zero LLM cost) re-tiers everything: active → fading → archived. Archived memories stop polluting retrieval but remain recoverable.

**Retrieval is rehearsal.** Accessing a memory strengthens it, exactly like human recall. Corrections get double weight — when a user says "no, that's wrong," that moment persists longer than neutral facts, because mistakes are the most expensive thing to repeat.

**Patterns harden into habits.** When the same preference gets reinforced past a threshold, the system generates a proposal: *"You've preferred TypeScript in 7 recent sessions — make it permanent?"* Approve it, and it becomes a standing instruction injected into every future prompt, forever exempt from decay.

**And the trust gate — the security decision I'd defend in any review:** the agent can *propose* permanent instructions, but only a human can approve them. There is no code path — none — by which an agent writes an active instruction. Memory poisoning (planting an instruction-shaped "fact" that executes weeks later) is a top-listed agentic risk for 2026, with published injection success rates above 95% against naive memory systems. Memgram additionally sanitizes every recalled memory before injection: a memory can state facts, but it cannot issue orders.

## Boring on purpose

The infrastructure thesis from the January article survived contact with reality almost untouched: **one Postgres for everything.** pgvector for similarity, plain SQL for the rest, an optional Apache AGE graph layer, row-level security for tenant isolation implemented once at the engine level. No dedicated vector database, no graph database, no LangChain — the entire "agent framework" is a base class of about sixty lines: a prompt, a JSON response, a retry loop.

What reality *added* is the unglamorous 80%: a transactional outbox so a queue outage can't lose a memory; a durable Streams queue where a crashed worker's jobs get reclaimed by its siblings; an anti-hallucination gate that fact-checks every extracted memory against the transcript before storing it (a false memory poisons every future prompt — worse than no memory); Prometheus metrics; property-based tests on the decay math. 154 automated checks across eight suites. The repo runs with `docker compose up`.

Total background cost: roughly $0.30 per active user per month on the default models — or about $0.05 if you point the extraction brain at DeepSeek or Gemini Flash-Lite, which is one environment variable.

## The benchmark — including where I lose

Vendor benchmarks are usually theater, so here are the rules I set myself: same conversations, same questions, same judge for every system; raw per-question results published; the harness is in the repo so you can rerun everything with one command; and the losses ship with the wins.

On the multi-session recall eval (4 personas, 24 questions across fact recall, fact *updates*, preferences, and must-abstain traps): **Memgram 96%, Mem0 96%.** A tie with the 48,000-star, $24M-funded category leader is a result I'll take at v0.1 — but a tie is a tie, and the details matter more:

- Where Mem0 stumbled: it missed a stated allergy — a recall miss on exactly the kind of fact you least want missed.
- Where I stumbled: a fact *update* ("promoted to tech lead") where my system retained the stale answer. Fact supersession is genuinely hard — my first contradiction-handling design made this axis *worse* (80% → 20%) by over-archiving correct facts, and I shelved it and rebuilt it around bounded per-fact operations. That redesign ships in the repo behind a flag until it beats the baseline on the eval. That's what "the losses ship too" means in practice.

The repo also includes a public leaderboard harness with adapters for Zep, Letta, LangMem, and a no-memory control — because a memory layer that can't beat "just paste the whole transcript" on something isn't earning its keep. If you maintain one of those systems and my adapter mis-uses your SDK, the PR button is right there.

## What it is honestly not, yet

No TypeScript SDK. Temporal reasoning is weaker than Zep's dedicated graph (that's their moat, and credit to them). Scale numbers are single-machine. Python only. And the final point of engineering credibility — years of hostile production traffic — cannot be built, only earned, which is precisely why this is launching now instead of after another six months of private polish.

## Your memory belongs to you

One more thing, because it fell out of the self-hosting decision and became my favorite feature: **memory is portable.** One endpoint exports everything your agent knows about a user as JSON; one endpoint imports it anywhere else — re-embedded on arrival, so it even works across different embedding models. Your memory is a file you can carry, not a moat a vendor keeps you inside. GDPR export and hard-delete came free with the same design.

There's a dashboard too — see every memory, its decay tier, its provenance, why it was recalled; edit instructions; approve or dismiss the agent's habit proposals. Memory you can't inspect is memory you can't trust.

## Try it

```bash
git clone https://github.com/YOUR-ORG/memgram && cd memgram
cp .env.example .env        # add your OpenAI key (or MEMGRAM_FAKE_LLM=1 for zero-cost)
docker compose up           # Postgres + Valkey + API + worker
pip install -e .
python examples/demo_persistence.py
```

Thirty seconds later your agent remembers something across a process restart, and you can watch the memory form in the dashboard.

I'm looking for ten developers to break this. File issues, run the leaderboard against your favorite memory system, tell me the decay defaults are wrong for your use case — that's exactly the data no benchmark can give me. The repo is [github.com/YOUR-ORG/memgram](https://github.com/YOUR-ORG/memgram), MIT-licensed, and everything in this article — the eval, the harness, the demo — is reproducible from it.

Six months ago I wrote that agent memory was a missing layer. It isn't missing anymore. Now help me find out what I got wrong.

---

*If you build agents and this resonates, a star on the repo genuinely helps other people find it. If you maintain a competing memory system — the leaderboard adapters await your corrections, sincerely.*
