import math
from scripts.benchmark_beir import recall_at_k, ndcg_at_k, reciprocal_rank


def test_recall_at_k():
    assert recall_at_k(["d3", "d1", "d2"], {"d1", "d2"}, 3) == 1.0
    assert recall_at_k(["d3", "d1", "d9"], {"d1", "d2"}, 3) == 0.5


def test_ndcg_perfect_is_1():
    assert ndcg_at_k(["d1", "d2", "d3"], {"d1", "d2"}, 10) == 1.0


def test_ndcg_rank_sensitive():
    good = ndcg_at_k(["d1", "x", "y"], {"d1"}, 10)
    worse = ndcg_at_k(["x", "y", "d1"], {"d1"}, 10)
    assert good > worse
    assert math.isclose(good, 1.0)


def test_ndcg_empty_relevant_is_0():
    assert ndcg_at_k(["d1"], set(), 10) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "d1", "y"], {"d1"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"d1"}) == 0.0
