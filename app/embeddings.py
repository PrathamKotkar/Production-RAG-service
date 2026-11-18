from __future__ import annotations

import re
import zlib
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from .config import Settings


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class STEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)


_TOK = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic feature-hashing embedder. Offline/CI only: lexical, not semantic."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = _TOK.findall(t.lower())
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for f in feats:
                h = zlib.crc32(f.encode())
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def build_embedder(s: Settings) -> Embedder:
    if s.embed_backend == "hash":
        return HashEmbedder(s.embed_dim)
    emb = STEmbedder(s.embed_model, s.embed_batch_size)
    if emb.dim != s.embed_dim:
        raise RuntimeError(
            f"Model dim {emb.dim} != EMBED_DIM {s.embed_dim}; update sql/init.sql and .env"
        )
    return emb


class QueryEmbedder:
    """Wraps an Embedder with an in-process LRU for repeated queries."""

    def __init__(self, embedder: Embedder, maxsize: int = 4096):
        self.embedder = embedder
        self._cached = lru_cache(maxsize=maxsize)(self._embed_one)

    def _embed_one(self, q: str) -> np.ndarray:
        return self.embedder.encode([q])[0]

    def __call__(self, q: str) -> np.ndarray:
        return self._cached(q)
