"""Ingestion worker: `python -m app.worker`"""
from __future__ import annotations

import asyncio
import logging
import signal

from .cache import Cache
from .config import get_settings
from .db import create_pool
from .embeddings import build_embedder
from .ingest import consumer, requeue_stuck
from redis.asyncio import Redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


async def main():
    s = get_settings()
    pool = await create_pool(s)
    redis = Redis.from_url(s.redis_url)
    embedder = build_embedder(s)
    cache = Cache(redis, s.cache_ttl_seconds, s.cache_enabled)

    stuck = await requeue_stuck(redis)
    if stuck:
        logging.info("requeued %d stuck messages", stuck)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    logging.info("worker started with %d consumers", s.ingest_consumers)
    await asyncio.gather(
        *[consumer(f"c{i}", pool, redis, embedder, cache, s, stop) for i in range(s.ingest_consumers)]
    )
    await pool.close()
    await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
