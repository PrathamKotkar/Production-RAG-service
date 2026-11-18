"""Reciprocal Rank Fusion and helpers. Pure functions."""
from __future__ import annotations

import re
from typing import Hashable, Sequence

_TOKEN = re.compile(r"[A-Za-z0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be by for from has have he her his how i in is it its of on or
    she that the their them they this to was were what when where which who whom why will
    with you your do does did can could would should about into than then there these those
    been being also not no but if so such""".split()
)


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Hashable]], k: int = 60
) -> list[tuple[Hashable, float]]:
    """Fuse ranked id lists. score(id) = sum_i 1 / (k + rank_i(id)), rank starting at 1."""
    scores: dict[Hashable, float] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))


def build_or_tsquery(query: str) -> str | None:
    """Natural-language question -> 'a | b | c' for to_tsquery (alnum tokens only, so safe)."""
    toks = [t.lower() for t in _TOKEN.findall(query)]
    toks = [t for t in dict.fromkeys(toks) if len(t) > 1 and t not in STOPWORDS]
    return " | ".join(toks) if toks else None


def dedupe_preserve_order(items: Sequence[Hashable]) -> list[Hashable]:
    return list(dict.fromkeys(items))
