from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .retrieval.fusion import STOPWORDS

if TYPE_CHECKING:
    from .config import Settings

SYSTEM = (
    "You answer questions using ONLY the numbered context passages. "
    "Cite passages inline like [1] or [2]. If the context does not contain the answer, "
    "reply exactly: I don't know based on the provided context. Be concise (1-3 sentences)."
)


def build_prompt(query: str, passages: list[str]) -> str:
    ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    return f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"


_SENT = re.compile(r"(?<=[.!?])\s+")
_TOK = re.compile(r"[a-z0-9]+")


def extractive_answer(query: str, passages: list[str], n_sentences: int = 2) -> str:
    """Offline fallback: pick the sentences with the highest query-term overlap."""
    q = {t for t in _TOK.findall(query.lower()) if t not in STOPWORDS}
    scored = []
    for pi, p in enumerate(passages, start=1):
        for s in _SENT.split(p):
            toks = set(_TOK.findall(s.lower()))
            if toks:
                scored.append((len(q & toks) / (len(toks) ** 0.5), pi, s.strip()))
    scored.sort(key=lambda x: -x[0])
    if not scored or scored[0][0] == 0:
        return "I don't know based on the provided context."
    return " ".join(f"{s} [{pi}]" for _, pi, s in scored[:n_sentences])


class Generator:
    def __init__(self, s: Settings):
        self.s = s
        self.client = None
        if s.anthropic_api_key:
            from anthropic import AsyncAnthropic

            self.client = AsyncAnthropic(api_key=s.anthropic_api_key)

    @property
    def backend(self) -> str:
        return "llm" if self.client else "extractive"

    async def answer(self, query: str, passages: list[str]) -> str:
        if not self.client:
            return extractive_answer(query, passages)
        msg = await self.client.messages.create(
            model=self.s.llm_model,
            max_tokens=self.s.llm_max_tokens,
            temperature=0,
            system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(query, passages)}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()
