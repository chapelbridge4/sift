from scripts.benchmark_beir import recall_at_k


def test_recall_at_k_basic():
    retrieved = ["d3", "d1", "d9", "d2", "d7"]
    relevant = {"d1", "d2"}
    assert recall_at_k(retrieved, relevant, k=5) == 1.0
    assert recall_at_k(retrieved, relevant, k=2) == 0.5
    assert recall_at_k(retrieved, relevant, k=1) == 0.0


def test_recall_empty_relevant():
    assert recall_at_k(["d1"], set(), k=1) == 0.0
