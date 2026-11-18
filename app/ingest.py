"""Async ingestion: API enqueues -> Redis reliable queue -> worker chunks/embeds/indexes."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid

from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis

from .cache import Cache
from .chunking import chunk_text
from .config import Settings
from .embeddings import Embedder

log = logging.getLogger("ingest")

QUEUE = "ingest:queue"
PROCESSING = "ingest:processing"
DEAD = "ingest:dead"


async def enqueue(pool: AsyncConnectionPool, redis: Redis, docs: list[dict], job_id: str | None = None) -> str:
    """Persist raw documents (status=pending) and push a message for the workers."""
    job_id = job_id or uuid.uuid4().hex
    async with pool.connection() as conn:
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO ingest_jobs (id, total) VALUES (%s, %s)", (job_id, len(docs))
            )
            async with conn.cursor() as cur:
                await cur.executemany(
                    """INSERT INTO documents (id, title, body, metadata, status)
                       VALUES (%s, %s, %s, %s::jsonb, 'pending')
                       ON CONFLICT (id) DO UPDATE
                         SET title = EXCLUDED.title, body = EXCLUDED.body,
                             metadata = EXCLUDED.metadata, status = 'pending'""",
                    [(d["id"], d.get("title", ""), d["body"], json.dumps(d.get("metadata", {}))) for d in docs],
                )
    await redis.rpush(QUEUE, json.dumps({"job_id": job_id, "doc_ids": [d["id"] for d in docs]}))
    return job_id


async def process_message(
    pool: AsyncConnectionPool, embedder: Embedder, cache: Cache, s: Settings, msg: dict
) -> int:
    ids: list[str] = msg["doc_ids"]
    job_id: str = msg["job_id"]
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE ingest_jobs SET status='running', updated_at=now() WHERE id=%s AND status='queued'",
            (job_id,),
        )
        cur = await conn.execute("SELECT id, title, body FROM documents WHERE id = ANY(%s)", (ids,))
        docs = await cur.fetchall()

    pieces: list[tuple[str, int, str]] = []
    for doc_id, _title, body in docs:
        for i, t in enumerate(chunk_text(body, s.chunk_words, s.chunk_overlap)):
            pieces.append((doc_id, i, t))
    titles = {d[0]: d[1] for d in docs}

    # Embed "title: chunk" in a thread so the event loop stays responsive
    embed_inputs = [f"{titles[d]}: {t}" if titles[d] else t for d, _, t in pieces]
    vectors = await asyncio.to_thread(embedder.encode, embed_inputs) if pieces else []

    async with pool.connection() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM chunks WHERE doc_id = ANY(%s)", (ids,))  # idempotent re-ingest
            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO chunks (doc_id, idx, text, embedding) VALUES (%s, %s, %s, %s)",
                    [(d, i, t, v) for (d, i, t), v in zip(pieces, vectors)],
                )
            await conn.execute("UPDATE documents SET status='indexed' WHERE id = ANY(%s)", (ids,))
            await conn.execute(
                """UPDATE ingest_jobs
                   SET done = done + %s, updated_at = now(),
                       status = CASE WHEN done + %s >= total THEN 'done' ELSE 'running' END
                   WHERE id = %s""",
                (len(ids), len(ids), job_id),
            )
    await cache.bump_version()
    return len(pieces)


async def _fail(pool: AsyncConnectionPool, job_id: str, err: str) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE ingest_jobs SET status='failed', error=%s, updated_at=now() WHERE id=%s",
            (err[:500], job_id),
        )


async def consumer(name: str, pool, redis: Redis, embedder, cache: Cache, s: Settings, stop: asyncio.Event):
    """At-least-once consumer: BLMOVE queue->processing, LREM on success, DLQ on failure."""
    while not stop.is_set():
        raw = await redis.blmove(QUEUE, PROCESSING, 2, "LEFT", "RIGHT")
        if raw is None:
            continue
        msg = json.loads(raw)
        try:
            n = await process_message(pool, embedder, cache, s, msg)
            log.info("%s job=%s docs=%d chunks=%d", name, msg["job_id"], len(msg["doc_ids"]), n)
        except Exception as e:  # noqa: BLE001
            log.exception("%s job=%s failed", name, msg["job_id"])
            await _fail(pool, msg["job_id"], repr(e))
            await redis.rpush(DEAD, raw)
        finally:
            await redis.lrem(PROCESSING, 1, raw)


async def requeue_stuck(redis: Redis) -> int:
    """On worker start, move anything left in PROCESSING (crashed worker) back to the queue."""
    n = 0
    while await redis.lmove(PROCESSING, QUEUE, "LEFT", "RIGHT"):
        n += 1
    return n
