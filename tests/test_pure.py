import math

import pytest

from app.chunking import chunk_text
from app.generation import extractive_answer
from app.keys import make_key, normalize_query
from app.retrieval.fusion import build_or_tsquery, rrf_fuse
from eval.metrics import (
    answer_contains_gold,
    hit_at_k,
    lexical_faithfulness,
    mrr,
    ndcg_at_k,
    recall_at_k,
    split_claims,
)


# ---- chunking
def test_chunk_short_text_single_chunk():
    assert chunk_text("a b c", 10, 2) == ["a b c"]


def test_chunk_overlap_and_coverage():
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_text(" ".join(words), chunk_words=30, overlap=10)
    assert all(len(c.split()) <= 30 for c in chunks)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]  # overlap preserved
    assert chunks[-1].split()[-1] == "w99"  # nothing dropped at the tail


def test_chunk_validation():
    with pytest.raises(ValueError):
        chunk_text("x", 10, 10)
    assert chunk_text("   ", 10, 2) == []


# ---- fusion
def test_rrf_prefers_items_in_both_lists():
    fused = dict(rrf_fuse([["a", "b", "c"], ["c", "d", "a"]], k=60))
    assert fused["a"] > fused["b"] and fused["c"] > fused["b"] and fused["a"] > fused["d"]
    assert math.isclose(fused["a"], 1 / 61 + 1 / 63)


def test_rrf_order_is_deterministic():
    r = rrf_fuse([["x", "y"], ["y", "x"]])
    assert [i for i, _ in r] == ["x", "y"]  # tie broken by id


def test_or_tsquery_drops_stopwords_and_sanitizes():
    q = build_or_tsquery("What is the capital of France? '; DROP TABLE x;--")
    assert q == "capital | france | drop | table"
    assert build_or_tsquery("the of and") is None


# ---- cache keys
def test_cache_key_normalizes_and_versions():
    a = make_key("ask", "  Who  WROTE it? ", {"k": 5}, "1")
    assert a == make_key("ask", "who wrote it?", {"k": 5}, "1")
    assert a != make_key("ask", "who wrote it?", {"k": 5}, "2")  # index bump invalidates
    assert a != make_key("ask", "who wrote it?", {"k": 6}, "1")
    assert normalize_query(" A   b ") == "a b"


# ---- retrieval metrics
def test_recall_hit_mrr_ndcg():
    ranked = ["d1", "d1", "d2", "d3", "d4"]  # duplicate chunks of d1 collapse to one doc
    rel = {"d3"}
    assert recall_at_k(ranked, rel, 2) == 0.0
    assert recall_at_k(ranked, rel, 3) == 1.0
    assert hit_at_k(ranked, rel, 3) == 1.0
    assert mrr(ranked, rel) == pytest.approx(1 / 3)
    assert ndcg_at_k(ranked, rel, 5) == pytest.approx(1 / math.log2(4))
    assert ndcg_at_k(["d3"], rel, 5) == 1.0
    assert recall_at_k([], set(), 5) == 0.0


# ---- answer metrics
def test_answer_contains_gold_normalization():
    assert answer_contains_gold("It was built in the year 1889.", ["1889"]) == 1.0
    assert answer_contains_gold("Gustave Eiffel, probably.", ["The Gustave Eiffel"]) == 1.0
    assert answer_contains_gold("No idea", ["Paris"]) == 0.0


def test_faithfulness_lexical():
    ctx = ["The Eiffel Tower was completed in 1889 in Paris."]
    assert lexical_faithfulness("The Eiffel Tower was completed in 1889 [1].", ctx) == 1.0
    assert lexical_faithfulness("The tower was designed by aliens from Mars.", ctx) == 0.0
    assert lexical_faithfulness("I don't know based on the provided context.", ctx) == 1.0
    claims = split_claims("Short. This is a real claim [1]. Another real claim here.")
    assert len(claims) == 2  # "Short." is dropped (<3 words); citation markers are stripped
    assert all("[" not in c for c in claims)


def test_extractive_answer_picks_relevant_sentence():
    passages = [
        "Paris is a city. The Eiffel Tower was completed in 1889.",
        "Bananas are yellow and grow in bunches.",
    ]
    ans = extractive_answer("When was the Eiffel Tower completed?", passages, 1)
    assert "1889" in ans and "[1]" in ans
    assert "don't know" in extractive_answer("quantum chromodynamics lattice", ["Bananas are yellow."])
