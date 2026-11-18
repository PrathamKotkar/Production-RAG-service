from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from psycopg_pool import AsyncConnectionPool

from ..config import Settings
from ..embeddings import QueryEmbedder
from .fusion import build_or_tsquery, rrf_fuse
from .rerank import Reranker

MODES = ("vector", "fts", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    idx: int
    title: str
    text: str
    score: float
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "idx": self.idx,
            "title": self.title,
            "text": self.text,
            "score": round(self.score, 6),
            "sources": self.sources,
        }


_VECTOR_SQL = """
SELECT c.id, c.doc_id, c.idx, d.title, c.text
FROM chunks c JOIN documents d ON d.id = c.doc_id
ORDER BY c.embedding <=> %s
LIMIT %s
"""

_FTS_SQL = """
SELECT c.id, c.doc_id, c.idx, d.title, c.text
FROM chunks c
JOIN documents d ON d.id = c.doc_id,
     to_tsquery('english', %s) q
WHERE c.tsv @@ q
ORDER BY ts_rank_cd(c.tsv, q) DESC
LIMIT %s
"""


class Retriever:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        settings: Settings,
        query_embedder: QueryEmbedder,
        reranker: Reranker | None,
    ):
        self.pool = pool
        self.s = settings
        self.embed = query_embedder
        self.reranker = reranker

    async def _vector(self, query: str, n: int) -> list[Hit]:
        qvec = await asyncio.to_thread(self.embed, query)
        async with self.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL hnsw.ef_search = {int(max(self.s.hnsw_ef_search, n))}")
                cur = await conn.execute(_VECTOR_SQL, (qvec, n))
                rows = await cur.fetchall()
        return [Hit(r[0], r[1], r[2], r[3], r[4], 0.0, ["vector"]) for r in rows]

    async def _fts(self, query: str, n: int) -> list[Hit]:
        tsq = build_or_tsquery(query)
        if tsq is None:
            return []
        async with self.pool.connection() as conn:
            cur = await conn.execute(_FTS_SQL, (tsq, n))
            rows = await cur.fetchall()
        return [Hit(r[0], r[1], r[2], r[3], r[4], 0.0, ["fts"]) for r in rows]

    @staticmethod
    def _rank_score(hits: list[Hit]) -> list[Hit]:
        # Score by rank for single-retriever modes (monotonic, keeps API uniform)
        for i, h in enumerate(hits, start=1):
            h.score = 1.0 / i
        return hits

    async def search(self, query: str, k: int = 5, mode: str = "hybrid_rerank") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        n = max(self.s.candidate_k, k)

        if mode == "vector":
            return self._rank_score(await self._vector(query, n))[:k]
        if mode == "fts":
            return self._rank_score(await self._fts(query, n))[:k]

        vec, fts = await asyncio.gather(self._vector(query, n), self._fts(query, n))
        by_id: dict[int, Hit] = {}
        for h in vec + fts:
            if h.chunk_id in by_id:
                by_id[h.chunk_id].sources = sorted(set(by_id[h.chunk_id].sources + h.sources))
            else:
                by_id[h.chunk_id] = h
        fused = rrf_fuse(
            [[h.chunk_id for h in vec], [h.chunk_id for h in fts]], k=self.s.rrf_k
        )
        ordered: list[Hit] = []
        for cid, score in fused:
            h = by_id[cid]
            h.score = score
            ordered.append(h)

        if mode == "hybrid" or self.reranker is None or not self.s.rerank_enabled:
            return ordered[:k]

        top = ordered[: max(self.s.rerank_top_n, k)]
        scores = await asyncio.to_thread(
            self.reranker.score, query, [f"{h.title}. {h.text}" for h in top]
        )
        for h, sc in zip(top, scores):
            h.score = sc
        top.sort(key=lambda h: -h.score)
        return top[:k]
