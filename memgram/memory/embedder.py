"""Embeddings with an in-process cache.

Big picture: the embedder is an interface, not a vendor. Three backends, chosen
by MEMGRAM_EMBEDDER (openai | local | fake):

  - openai : text-embedding-3-small (1536d). Best quality, but a ~300ms network
             round-trip on the hot path — this is what breaks the latency promise.
  - local  : fastembed / ONNX (bge-small-en-v1.5, 384d) loaded INTO the process.
             No network hop; ~3-8ms/query on CPU. The hot path stays single-digit
             ms. Lower quality than OpenAI — measure with bench/quality.
  - fake   : deterministic hash (tests / zero-infra dev).

IMPORTANT: the API (query) and worker (stored memories) must use the SAME backend
or cosine similarity is meaningless. The embedding dimension is fixed at table
creation — switching backends means re-embedding (set MEMGRAM_EMBED_DIMS to match
BEFORE first migrate: 1536 for openai, 384 for the local bge-small default).
"""
import asyncio
import hashlib
import math
import os

DIMS = int(os.environ.get("MEMGRAM_EMBED_DIMS", "1536"))
_MODEL = os.environ.get("MEMGRAM_EMBED_MODEL", "text-embedding-3-small")
_LOCAL_MODEL = os.environ.get("MEMGRAM_LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")


class FakeEmbedder:
    """Deterministic bag-of-words embedding: each token hashes to a fixed
    pseudo-vector; a text embeds as the normalized sum. Identical strings get
    identical vectors (dedup works) and RELATED sentences get RELATED vectors
    ("lives in Berlin" ~ "lives in Munich"), so similarity-driven behaviour —
    dedup, contradiction candidates, ranked retrieval — is realistic with zero
    network calls. Not a semantic model: paraphrases without shared tokens
    won't match."""

    _token_cache: dict = {}

    def _token_vec(self, tok: str) -> list[float]:
        if tok not in self._token_cache:
            h = hashlib.sha256(tok.encode()).digest()
            self._token_cache[tok] = [
                ((h[(i * 7 + 3) % 32] * 31 + i * 13) % 997 - 498.0) for i in range(DIMS)]
        return self._token_cache[tok]

    async def embed(self, text: str) -> list[float]:
        toks = [t for t in "".join(
            c if c.isalnum() else " " for c in text.lower()).split() if t]
        if not toks:
            toks = ["empty"]
        vals = [0.0] * DIMS
        for tok in set(toks):
            tv = self._token_vec(tok)
            for i in range(DIMS):
                vals[i] += tv[i]
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


class LocalEmbedder:
    """In-process ONNX embeddings via fastembed — no network hop. The model is
    loaded once (here / at FastAPI lifespan startup) and inference is offloaded to
    a thread so it never blocks the async event loop."""

    def __init__(self):
        from fastembed import TextEmbedding
        self._model = TextEmbedding(_LOCAL_MODEL)
        self._cache: dict[str, list[float]] = {}

    def _embed_sync(self, text: str) -> list[float]:
        vec = next(iter(self._model.embed([text])))  # generator -> first vector
        return [float(x) for x in vec]

    async def embed(self, text: str) -> list[float]:
        key = hashlib.sha256(text.encode()).hexdigest()
        if key not in self._cache:
            if len(self._cache) > 10_000:
                self._cache.clear()
            self._cache[key] = await asyncio.to_thread(self._embed_sync, text)
        return self._cache[key]


def get_embedder():
    # FAKE wins unconditionally (tests / zero-infra dev).
    if os.environ.get("MEMGRAM_FAKE_LLM"):
        return FakeEmbedder()
    backend = os.environ.get("MEMGRAM_EMBEDDER", "").lower()
    if backend == "local":
        return LocalEmbedder()
    if backend == "openai":
        return OpenAIEmbedder()
    # default: openai when a key is present, else fake (keeps no-infra dev working)
    return OpenAIEmbedder() if os.environ.get("OPENAI_API_KEY") else FakeEmbedder()


def to_pgvector(v: list[float]) -> str:
    """asyncpg-friendly literal; cast with ::vector in SQL."""
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"
