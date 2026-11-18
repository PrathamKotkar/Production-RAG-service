"""Word-window chunking with overlap. Pure function, no deps."""
from __future__ import annotations


def chunk_text(text: str, chunk_words: int = 180, overlap: int = 30) -> list[str]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if not 0 <= overlap < chunk_words:
        raise ValueError("overlap must be in [0, chunk_words)")
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    step = chunk_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        chunks.append(" ".join(window))
        if start + chunk_words >= len(words):
            break
    return chunks
