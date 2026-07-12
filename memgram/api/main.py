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
import time
from contextlib import asynccontextmanager

import asyncpg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from memgram.obs import (CONTENT_TYPE_LATEST, DLQ_DEPTH, QUEUE_DEPTH,
                         REQUEST_SECONDS, generate_latest, new_request_id,
                         setup_logging)
from memgram.api.routes.context import router as context_router
from memgram.api.routes.findings import router as findings_router
from memgram.api.routes.settings import router as settings_router
from memgram.api.routes.ingest import router as ingest_router
from memgram.api.routes.usage_route import router as usage_router
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
    setup_logging()
    app.state.pool = await asyncpg.create_pool(
        os.environ["DATABASE_URL"], min_size=1,
        max_size=int(os.environ.get("MEMGRAM_POOL_MAX", "10")))
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
app.include_router(context_router, prefix="/v1/context", dependencies=auth)
app.include_router(usage_router, prefix="/v1/usage", dependencies=auth)
app.include_router(findings_router, prefix="/v1/findings", dependencies=auth)
app.include_router(settings_router, prefix="/v1/settings", dependencies=auth)
app.include_router(memory_router, prefix="/v1/memory", dependencies=auth)
app.include_router(proposals_router, prefix="/v1/proposals", dependencies=auth)


@app.get("/health")
async def health(request: Request):
    async with request.app.state.pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    q = request.app.state.queue
    return {"status": "ok", "queue_depth": q.depth(), "dead_letter": q.dlq_depth()}


# -- observability ------------------------------------------------------------
from fastapi import Response  # noqa: E402


@app.middleware("http")
async def _obs_middleware(request: Request, call_next):
    rid = new_request_id(request.headers.get("x-request-id"))
    t0 = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    REQUEST_SECONDS.labels(
        route=getattr(route, "path", request.url.path),
        method=request.method, status=str(response.status_code),
    ).observe(time.perf_counter() - t0)
    response.headers["x-request-id"] = rid
    return response


@app.get("/metrics")
async def metrics(request: Request):
    q = request.app.state.queue
    try:
        QUEUE_DEPTH.set(q.depth())
        DLQ_DEPTH.set(q.dlq_depth())
    except Exception:
        pass
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
