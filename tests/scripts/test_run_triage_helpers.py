"""Unit tests for the pure helper in scripts/run_triage.py.

Only the deterministic ``rerank_to_doc_ids`` mapper is tested here; the
model-loading and async-pipeline parts of the runner are integration concerns
exercised by running the script, not by CI.
"""

from __future__ import annotations

from scripts.run_triage import rerank_to_doc_ids


def test_rerank_to_doc_ids_preserves_order():
    reranked = [
        {"doc_id": "c", "rerank_score": 0.9, "text": "..."},
        {"doc_id": "a", "rerank_score": 0.5, "text": "..."},
        {"doc_id": "b", "rerank_score": 0.1, "text": "..."},
    ]
    assert rerank_to_doc_ids(reranked) == ["c", "a", "b"]


def test_rerank_to_doc_ids_empty():
    assert rerank_to_doc_ids([]) == []


def test_rerank_to_doc_ids_skips_missing_ids():
    reranked = [
        {"doc_id": "x", "rerank_score": 0.9},
        {"rerank_score": 0.5},  # no doc_id -> skipped
        {"doc_id": None, "rerank_score": 0.4},  # explicit None -> skipped
        {"doc_id": "y", "rerank_score": 0.1},
    ]
    assert rerank_to_doc_ids(reranked) == ["x", "y"]
