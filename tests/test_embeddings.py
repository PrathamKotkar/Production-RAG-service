import numpy as np

from app.embeddings import HashEmbedder


def test_hash_embedder_deterministic_unit_norm():
    e = HashEmbedder(384)
    a = e.encode(["the quick brown fox"])
    b = e.encode(["the quick brown fox"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)


def test_hash_embedder_similarity_ordering():
    e = HashEmbedder(384)
    q, near, far = e.encode(["eiffel tower paris height", "the eiffel tower in paris is tall", "banana bread recipe"])
    assert q @ near > q @ far
