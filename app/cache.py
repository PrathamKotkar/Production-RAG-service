from __future__ import annotations

import json

from redis.asyncio import Redis

from .keys import make_key

VERSION_KEY = "rag:index_version"
HITS = "rag:stats:hits"
MISSES = "rag:stats:misses"


class Cache:
    """Redis response cache. The index version is part of every key, so any ingest
    (which bumps the version) invalidates all cached answers without scanning."""

    def __init__(self, redis: Redis, ttl: int, enabled: bool = True):
        self.r = redis
        self.ttl = ttl
        self.enabled = enabled

    async def version(self) -> str:
        v = await self.r.get(VERSION_KEY)
        return v.decode() if isinstance(v, bytes) else (v or "0")

    async def bump_version(self) -> None:
        await self.r.incr(VERSION_KEY)

    async def key(self, kind: str, query: str, params: dict) -> str:
        return make_key(kind, query, params, await self.version())

    async def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        raw = await self.r.get(key)
        if raw is None:
            await self.r.incr(MISSES)
            return None
        await self.r.incr(HITS)
        return json.loads(raw)

    async def set(self, key: str, value: dict) -> None:
        if self.enabled:
            await self.r.set(key, json.dumps(value), ex=self.ttl)

    async def stats(self) -> dict:
        h, m = await self.r.mget(HITS, MISSES)
        h, m = int(h or 0), int(m or 0)
        total = h + m
        return {"hits": h, "misses": m, "hit_rate": round(h / total, 4) if total else 0.0}

    async def reset_stats(self) -> None:
        await self.r.delete(HITS, MISSES)
