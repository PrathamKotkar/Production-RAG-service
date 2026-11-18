"""Retrieval and answer-quality metrics. Pure functions, no deps."""
from __future__ import annotations

import math
import re
import string
from typing import Sequence

_TOK = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to was were with".split()
)


def recall_at_k(ranked_doc_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k (distinct docs)."""
    if not relevant:
        return 0.0
    top = list(dict.fromkeys(ranked_doc_ids))[:k]
    return len(relevant & set(top)) / len(relevant)


def hit_at_k(ranked_doc_ids: Sequence[str], relevant: set[str], k: int) -> float:
    top = list(dict.fromkeys(ranked_doc_ids))[:k]
    return 1.0 if relevant & set(top) else 0.0


def mrr(ranked_doc_ids: Sequence[str], relevant: set[str], k: int = 10) -> float:
    for rank, d in enumerate(list(dict.fromkeys(ranked_doc_ids))[:k], start=1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_doc_ids: Sequence[str], relevant: set[str], k: int) -> float:
    top = list(dict.fromkeys(ranked_doc_ids))[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(top) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


_PUNCT = set(string.punctuation)


def _norm(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation and articles, collapse spaces."""
    s = "".join(ch for ch in s.lower() if ch not in _PUNCT)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def answer_contains_gold(answer: str, golds: Sequence[str]) -> float:
    """1.0 if any normalized gold answer string appears in the normalized answer."""
    a = _norm(answer)
    return 1.0 if any(_norm(g) and _norm(g) in a for g in golds) else 0.0


def split_claims(answer: str) -> list[str]:
    cleaned = re.sub(r"\[\d+\]", "", answer)
    parts = re.split(r"(?<=[.!?])\s+", cleaned.strip())
    return [p.strip() for p in parts if len(p.split()) >= 3]


def lexical_support(claim: str, context: str, threshold: float = 0.7) -> bool:
    """Cheap faithfulness proxy: >= threshold of the claim's content tokens appear in the context."""
    ctoks = {t for t in _TOK.findall(claim.lower()) if t not in _STOP}
    if not ctoks:
        return True
    have = set(_TOK.findall(context.lower()))
    return len(ctoks & have) / len(ctoks) >= threshold


def lexical_faithfulness(answer: str, contexts: Sequence[str]) -> float:
    """Fraction of answer claims supported by the retrieved context. 'I don't know' counts as faithful."""
    if "i don't know" in answer.lower():
        return 1.0
    claims = split_claims(answer)
    if not claims:
        return 0.0
    ctx = " ".join(contexts)
    return sum(lexical_support(c, ctx) for c in claims) / len(claims)
