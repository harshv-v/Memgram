"""Embeddings with an in-process cache.

Big picture: the embedder is an interface, not a vendor. OpenAI today;
swappable for a local model later without touching store/retriever code.
Set MEMGRAM_FAKE_LLM=1 (or run without OPENAI_API_KEY) to get a deterministic
fake — the full pipeline runs with zero external calls.
"""
import hashlib
import math
import os

DIMS = int(os.environ.get("MEMGRAM_EMBED_DIMS", "1536"))
_MODEL = os.environ.get("MEMGRAM_EMBED_MODEL", "text-embedding-3-small")


class FakeEmbedder:
    """Deterministic, normalized, content-sensitive. Similar strings do NOT
    get similar vectors (it's a hash) — tests that need similarity reuse
    exact strings."""

    async def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.lower().strip().encode()).digest()
        vals = [(h[i % 32] * 31 + i * 7) % 997 - 498.0 for i in range(DIMS)]
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        return [v / norm for v in vals]


class OpenAIEmbedder:
    def __init__(self):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI()
        self._cache: dict[str, list[float]] = {}

    async def embed(self, text: str) -> list[float]:
        key = hashlib.sha256(text.encode()).hexdigest()
        if key not in self._cache:
            r = await self._client.embeddings.create(model=_MODEL, input=text)
            if len(self._cache) > 10_000:  # crude bound; LRU later if it matters
                self._cache.clear()
            self._cache[key] = r.data[0].embedding
        return self._cache[key]


def get_embedder():
    if os.environ.get("MEMGRAM_FAKE_LLM") or not os.environ.get("OPENAI_API_KEY"):
        return FakeEmbedder()
    return OpenAIEmbedder()


def to_pgvector(v: list[float]) -> str:
    """asyncpg-friendly literal; cast with ::vector in SQL."""
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"
