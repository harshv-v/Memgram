"""Memgram backend API.

Run:
    DATABASE_URL=postgresql://... MEMGRAM_API_KEY=... uvicorn memgram.api.main:app
Env:
    DATABASE_URL        Postgres DSN (pgvector enabled)
    REDIS_URL           Redis/Valkey URL (default redis://localhost:6379/0)
    MEMGRAM_API_KEY      bearer token developers use (default mgram_dev_key)
    OPENAI_API_KEY      embeddings on the hot path (omit + MEMGRAM_FAKE_LLM=1 for dev)
"""
import os
from contextlib import asynccontextmanager

import asyncpg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from memgram.api.routes.ingest import router as ingest_router
from memgram.api.routes.instructions import router as instructions_router
from memgram.api.routes.memory import router as memory_router
from memgram.api.routes.proposals import router as proposals_router
from memgram.memory.embedder import get_embedder
from memgram.memory.retriever import Retriever
from memgram.memory.store import MemoryStore
from memgram.worker.queue import JobQueue

_bearer = HTTPBearer(auto_error=False)


async def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    expected = os.environ.get("MEMGRAM_API_KEY", "mgram_dev_key")
    if credentials is None or credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return credentials.credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(
        os.environ["DATABASE_URL"], min_size=1, max_size=10)
    app.state.embedder = get_embedder()
    app.state.store = MemoryStore(app.state.pool, app.state.embedder)
    app.state.retriever = Retriever(app.state.pool, app.state.embedder)
    app.state.queue = JobQueue()
    yield
    await app.state.pool.close()


app = FastAPI(title="Memgram", version="0.1.0", lifespan=lifespan)
app.add_middleware(  # dashboard runs on another port
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

auth = [Depends(require_api_key)]
app.include_router(instructions_router, prefix="/v1/instructions", dependencies=auth)
app.include_router(ingest_router, prefix="/v1/ingest", dependencies=auth)
app.include_router(memory_router, prefix="/v1/memory", dependencies=auth)
app.include_router(proposals_router, prefix="/v1/proposals", dependencies=auth)


@app.get("/health")
async def health(request: Request):
    async with request.app.state.pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    return {"status": "ok", "queue_depth": request.app.state.queue.depth()}
