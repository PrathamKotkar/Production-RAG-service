from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["vector", "fts", "hybrid", "hybrid_rerank"]


class DocumentIn(BaseModel):
    id: str
    title: str = ""
    body: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1, max_length=1000)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=50)
    mode: Mode = "hybrid_rerank"
    use_cache: bool = True
