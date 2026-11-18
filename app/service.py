"""Query orchestration: cache -> single-flight -> retrieve (-> generate) -> cache."""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Awaitable, Callable

from .bootstrap import Components


class QueryService:
    def __init__(self, c: Components):
        self.c = c
        self._inflight: dict[str, asyncio.Future] = {}
        self.latencies_ms: deque[float] = deque(maxlen=5000)  # per-process rolling window

    async def _cached(self, key: str, use_cache: bool, compute: Callable[[], Awaitable[dict]]):
        if use_cache:
            hit = await self.c.cache.get(key)
            if hit is not None:
                return hit, "HIT"
        # single-flight: concurrent identical misses share one computation (no stampede)
        if key in self._inflight:
            return await asyncio.shield(self._inflight[key]), "COALESCED"
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        try:
            result = await compute()
            if use_cache:
                await self.c.cache.set(key, result)
            fut.set_result(result)
            return result, "MISS"
        except Exception as e:
            fut.set_exception(e)
            fut.exception()  # mark retrieved
            raise
        finally:
            self._inflight.pop(key, None)

    async def search(self, query: str, k: int, mode: str, use_cache: bool = True):
        key = await self.c.cache.key("search", query, {"k": k, "mode": mode})

        async def compute():
            hits = await self.c.retriever.search(query, k=k, mode=mode)
            return {"query": query, "mode": mode, "hits": [h.to_dict() for h in hits]}

        return await self._cached(key, use_cache, compute)

    async def ask(self, query: str, k: int, mode: str, use_cache: bool = True):
        key = await self.c.cache.key("ask", query, {"k": k, "mode": mode, "gen": self.c.generator.backend})

        async def compute():
            hits = await self.c.retriever.search(query, k=k, mode=mode)
            answer = await self.c.generator.answer(query, [h.text for h in hits])
            return {
                "query": query,
                "answer": answer,
                "generator": self.c.generator.backend,
                "citations": [
                    {"n": i, "doc_id": h.doc_id, "title": h.title, "idx": h.idx}
                    for i, h in enumerate(hits, start=1)
                ],
                "contexts": [h.text for h in hits],
            }

        return await self._cached(key, use_cache, compute)

    def latency_summary(self) -> dict:
        xs = sorted(self.latencies_ms)
        if not xs:
            return {"count": 0}
        pick = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
        return {"count": len(xs), "p50_ms": round(pick(0.5), 1), "p95_ms": round(pick(0.95), 1), "p99_ms": round(pick(0.99), 1)}
