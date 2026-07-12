"""Usage accounting — every token the memory layer touches, per user.

Three kinds of consumption, all recorded to `usage_events`:
  llm:<Agent>   background brain calls (extractor/verify/reflection/...)
  injection     tokens Memgram ADDED to the developer's prompt (hot path)
  embedding     embedding calls (estimated from text length)

Costs are estimates from a built-in price table (per 1M tokens, USD) —
override any price with MEMGRAM_PRICE_<MODEL>="in,out" (dots/dashes -> _).
Recording is always best-effort: accounting must never break the pipeline.
"""
import logging
import os

from memgram.obs import Counter

logger = logging.getLogger("memgram.usage")

TOKENS_TOTAL = Counter("memgram_tokens_total",
                       "Tokens consumed/injected by the memory layer",
                       ["kind", "direction"])

# per 1M tokens (input, output), USD — estimates; override via env
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "deepseek-chat": (0.14, 0.28),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.15, 0.60),
    "claude-haiku-4-5": (0.25, 1.25),
    "claude-sonnet-4-6": (3.00, 15.00),
    "text-embedding-3-small": (0.02, 0.0),
}


def price_for(model: str | None) -> tuple[float, float]:
    if not model:
        return (0.0, 0.0)
    env = os.environ.get("MEMGRAM_PRICE_" + model.upper().replace("-", "_").replace(".", "_").replace("/", "_"))
    if env:
        try:
            i, o = env.split(",")
            return (float(i), float(o))
        except ValueError:
            pass
    for known, p in PRICES.items():
        if model.startswith(known):
            return p
    return (0.0, 0.0)


def estimate_cost(model: str | None, tokens_in: int, tokens_out: int) -> float:
    pi, po = price_for(model)
    return (tokens_in * pi + tokens_out * po) / 1_000_000


def norm_usage(usage) -> tuple[int, int]:
    """Normalize OpenAI (.prompt_tokens/.completion_tokens), Anthropic
    (.input_tokens/.output_tokens), or dict usage into (in, out)."""
    if usage is None:
        return (0, 0)
    def g(*names):
        for n in names:
            v = usage.get(n) if isinstance(usage, dict) else getattr(usage, n, None)
            if v is not None:
                return int(v)
        return 0
    return (g("prompt_tokens", "input_tokens", "in"),
            g("completion_tokens", "output_tokens", "out"))


async def record(pool, project_id: str, agent_id: str, user_id: str, kind: str,
                 model: str | None = None, tokens_in: int = 0, tokens_out: int = 0) -> None:
    """Best-effort insert + metrics. Never raises."""
    try:
        TOKENS_TOTAL.labels(kind=kind, direction="in").inc(tokens_in)
        TOKENS_TOTAL.labels(kind=kind, direction="out").inc(tokens_out)
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO usage_events
                   (project_id, agent_id, user_id, kind, model, tokens_in, tokens_out, cost_usd)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8)""",
                project_id, agent_id, user_id, kind, model,
                tokens_in, tokens_out, estimate_cost(model, tokens_in, tokens_out))
    except Exception as e:
        logger.debug("usage record skipped: %s", e)


def est_tokens(text: str) -> int:
    return len(text) // 4 if text else 0
