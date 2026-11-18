"""Builds the shared component graph for the API, worker, and eval harness."""
from __future__ import annotations

from dataclasses import dataclass

from redis.asyncio import Redis

from .cache import Cache
from .config import Settings, get_settings
from .db import create_pool
from .embeddings import QueryEmbedder, build_embedder
from .generation import Generator
from .retrieval.rerank import CrossEncoderReranker
from .retrieval.retriever import Retriever


@dataclass
class Components:
    settings: Settings
    pool: object
    redis: Redis
    embedder: object
    cache: Cache
    retriever: Retriever
    generator: Generator

    async def close(self):
        await self.pool.close()
        await self.redis.aclose()


async def build_components(s: Settings | None = None) -> Components:
    s = s or get_settings()
    pool = await create_pool(s)
    redis = Redis.from_url(s.redis_url)
    embedder = build_embedder(s)
    reranker = CrossEncoderReranker(s.rerank_model) if s.rerank_enabled else None
    cache = Cache(redis, s.cache_ttl_seconds, s.cache_enabled)
    retriever = Retriever(pool, s, QueryEmbedder(embedder), reranker)
    return Components(s, pool, redis, embedder, cache, retriever, Generator(s))
