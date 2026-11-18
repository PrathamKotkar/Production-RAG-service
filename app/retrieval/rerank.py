from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=384)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(x) for x in self.model.predict([(query, t) for t in texts], batch_size=32)]
