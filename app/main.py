from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response

from .bootstrap import build_components
from .ingest import DEAD, PROCESSING, QUEUE, enqueue
from .schemas import IngestRequest, QueryRequest
from .service import QueryService


@asynccontextmanager
async def lifespan(app: FastAPI):
    comps = await build_components()
    app.state.c = comps
    app.state.svc = QueryService(comps)
    # Warm models so the first real request isn't slow
    await comps.retriever.search("warmup", k=1, mode="hybrid_rerank")
    yield
    await comps.close()


app = FastAPI(title="Production RAG Service", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Response-Time-ms"] = f"{ms:.1f}"
    if request.url.path.startswith("/v1/") and request.method == "POST" and hasattr(request.app.state, "svc"):
        request.app.state.svc.latencies_ms.append(ms)
    return response


@app.get("/healthz")
async def healthz(request: Request):
    c = request.app.state.c
    async with c.pool.connection() as conn:
        await conn.execute("SELECT 1")
    await c.redis.ping()
    return {"status": "ok"}


@app.post("/v1/documents", status_code=202)
async def ingest(req: IngestRequest, request: Request):
    c = request.app.state.c
    job_id = await enqueue(c.pool, c.redis, [d.model_dump() for d in req.documents])
    return {"job_id": job_id, "queued": len(req.documents)}


@app.get("/v1/jobs/{job_id}")
async def job_status(job_id: str, request: Request):
    c = request.app.state.c
    async with c.pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, total, done, status, error, created_at, updated_at FROM ingest_jobs WHERE id=%s",
            (job_id,),
        )
        row = await cur.fetchone()
    if not row:
        raise HTTPException(404, "job not found")
    keys = ("id", "total", "done", "status", "error", "created_at", "updated_at")
    return {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in zip(keys, row)}


@app.post("/v1/search")
async def search(req: QueryRequest, request: Request, response: Response):
    result, state = await request.app.state.svc.search(req.query, req.k, req.mode, req.use_cache)
    response.headers["X-Cache"] = state
    return result


@app.post("/v1/ask")
async def ask(req: QueryRequest, request: Request, response: Response):
    result, state = await request.app.state.svc.ask(req.query, req.k, req.mode, req.use_cache)
    response.headers["X-Cache"] = state
    return result


@app.get("/metrics")
async def metrics(request: Request):
    c = request.app.state.c
    async with c.pool.connection() as conn:
        cur = await conn.execute(
            "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)"
        )
        docs, chunks = await cur.fetchone()
    return {
        "documents": docs,
        "chunks": chunks,
        "cache": await c.cache.stats(),
        "latency_this_process": request.app.state.svc.latency_summary(),
        "queue": {
            "pending": await c.redis.llen(QUEUE),
            "processing": await c.redis.llen(PROCESSING),
            "dead": await c.redis.llen(DEAD),
        },
    }


@app.post("/admin/cache/reset-stats")
async def reset_stats(request: Request):
    await request.app.state.c.cache.reset_stats()
    return {"ok": True}
