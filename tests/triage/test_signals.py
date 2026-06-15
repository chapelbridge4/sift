"""Tests for the deterministic span signals layer (app/triage/signals.py).

Written TDD-first: all tests must FAIL before signals.py exists, then PASS
after implementation.  No model loading, no network, no I/O.
"""

import pytest
from app.triage.signals import QueryTrace, Signals, compute_signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(doc_id: str, score: float, text: str = "") -> dict:
    d = {"doc_id": doc_id, "score": score}
    if text:
        d["text"] = text
    return d


# ---------------------------------------------------------------------------
# recall_hit / gold_rank
# ---------------------------------------------------------------------------

class TestRecallAndGoldRank:
    def test_gold_missing_from_retrieved(self):
        """Gold doc not in retrieved -> recall_hit=False, gold_rank=None."""
        trace = QueryTrace(
            query="what is qdrant",
            retrieved=[_doc("a", 0.9), _doc("b", 0.8), _doc("c", 0.7)],
            gold_ids={"gold_doc"},
            reranked=None,
            answer="Qdrant is a vector database.",
            top_k=3,
        )
        sig = compute_signals(trace)
        assert sig.recall_hit is False
        assert sig.gold_rank is None

    def test_gold_retrieved_at_rank_3(self):
        """Gold doc at position index 2 (rank 3) -> recall_hit=True, gold_rank=3."""
        trace = QueryTrace(
            query="what is qdrant",
            retrieved=[
                _doc("a", 0.9),
                _doc("b", 0.85),
                _doc("gold_doc", 0.8),
                _doc("d", 0.75),
                _doc("e", 0.7),
            ],
            gold_ids={"gold_doc"},
            reranked=None,
            answer="Qdrant is a vector database.",
            top_k=5,
        )
        sig = compute_signals(trace)
        assert sig.recall_hit is True
        assert sig.gold_rank == 3

    def test_gold_at_rank_1(self):
        """Gold doc at first position -> gold_rank=1."""
        trace = QueryTrace(
            query="q",
            retrieved=[_doc("gold", 0.99), _doc("other", 0.5)],
            gold_ids={"gold"},
            reranked=None,
            answer="answer",
            top_k=2,
        )
        sig = compute_signals(trace)
        assert sig.recall_hit is True
        assert sig.gold_rank == 1

    def test_empty_gold_ids(self):
        """No gold ids -> recall_hit=False, gold_rank=None (unknown)."""
        trace = QueryTrace(
            query="q",
            retrieved=[_doc("a", 0.9)],
            gold_ids=set(),
            reranked=None,
            answer="answer",
            top_k=1,
        )
        sig = compute_signals(trace)
        assert sig.recall_hit is False
        assert sig.gold_rank is None

    def test_multiple_gold_ids_best_rank(self):
        """With two gold docs, gold_rank is the best (lowest number) rank."""
        trace = QueryTrace(
            query="q",
            retrieved=[
                _doc("x", 0.9),
                _doc("gold2", 0.85),
                _doc("gold1", 0.8),
            ],
            gold_ids={"gold1", "gold2"},
            reranked=None,
            answer="answer",
            top_k=3,
        )
        sig = compute_signals(trace)
        assert sig.recall_hit is True
        assert sig.gold_rank == 2  # gold2 is at index 1 -> rank 2


# ---------------------------------------------------------------------------
# score_gap
# ---------------------------------------------------------------------------

class TestScoreGap:
    def test_score_gap_computed(self):
        """score_gap = top1_score - topk_score."""
        trace = QueryTrace(
            query="q",
            retrieved=[
                _doc("a", 0.9),
                _doc("b", 0.7),
                _doc("c", 0.5),
            ],
            gold_ids=set(),
            reranked=None,
            answer=None,
            top_k=3,
        )
        sig = compute_signals(trace)
        assert pytest.approx(sig.score_gap, abs=1e-6) == 0.9 - 0.5

    def test_score_gap_zero_with_one_result(self):
        """Only one retrieved doc -> score_gap=0.0."""
        trace = QueryTrace(
            query="q",
            retrieved=[_doc("a", 0.9)],
            gold_ids=set(),
            reranked=None,
            answer=None,
            top_k=1,
        )
        sig = compute_signals(trace)
        assert sig.score_gap == 0.0

    def test_score_gap_zero_with_empty_retrieved(self):
        """Empty retrieved list -> score_gap=0.0."""
        trace = QueryTrace(
            query="q",
            retrieved=[],
            gold_ids=set(),
            reranked=None,
            answer=None,
            top_k=5,
        )
        sig = compute_signals(trace)
        assert sig.score_gap == 0.0


# ---------------------------------------------------------------------------
# reranker_moved_gold_down
# ---------------------------------------------------------------------------

class TestRerankerMovedGoldDown:
    def test_reranker_pushes_gold_down(self):
        """Gold was rank 1 pre-rerank, then pushed to last -> True."""
        # retrieved order: gold, a, b, c, d  (gold at rank 1)
        # reranked order:  a, b, c, d, gold  (gold at rank 5)
        trace = QueryTrace(
            query="q",
            retrieved=[
                _doc("gold", 0.9),
                _doc("a", 0.85),
                _doc("b", 0.8),
                _doc("c", 0.75),
                _doc("d", 0.7),
            ],
            gold_ids={"gold"},
            reranked=["a", "b", "c", "d", "gold"],
            answer="answer",
            top_k=5,
        )
        sig = compute_signals(trace)
        assert sig.reranker_moved_gold_down is True

    def test_reranker_keeps_gold_up(self):
        """Gold was rank 3 pre-rerank, then stays at rank 1 post-rerank -> False."""
        trace = QueryTrace(
            query="q",
            retrieved=[
                _doc("a", 0.9),
                _doc("b", 0.85),
                _doc("gold", 0.8),
            ],
            gold_ids={"gold"},
            reranked=["gold", "a", "b"],
            answer="answer",
            top_k=3,
        )
        sig = compute_signals(trace)
        assert sig.reranker_moved_gold_down is False

    def test_no_reranked_list_is_false(self):
        """When reranked=None, the signal must be False (no reranker ran)."""
        trace = QueryTrace(
            query="q",
            retrieved=[_doc("gold", 0.9), _doc("a", 0.8)],
            gold_ids={"gold"},
            reranked=None,
            answer="answer",
            top_k=2,
        )
        sig = compute_signals(trace)
        assert sig.reranker_moved_gold_down is False

    def test_no_gold_ids_reranker_false(self):
        """Unknown gold -> reranker_moved_gold_down=False (can't evaluate)."""
        trace = QueryTrace(
            query="q",
            retrieved=[_doc("a", 0.9)],
            gold_ids=set(),
            reranked=["a"],
            answer="answer",
            top_k=1,
        )
        sig = compute_signals(trace)
        assert sig.reranker_moved_gold_down is False


# ---------------------------------------------------------------------------
# answer_present / answer_quality_ok
# ---------------------------------------------------------------------------

class TestAnswerQuality:
    def test_empty_answer_flags_invalid(self):
        """Empty answer -> answer_present=False, answer_quality_ok=False."""
        trace = QueryTrace(
            query="q",
            retrieved=[_doc("a", 0.9)],
            gold_ids=set(),
            reranked=None,
            answer="",
            top_k=1,
        )
        sig = compute_signals(trace)
        assert sig.answer_present is False
        assert sig.answer_quality_ok is False

    def test_none_answer_flags_invalid(self):
        """None answer -> answer_present=False, answer_quality_ok=False."""
        trace = QueryTrace(
            query="q",
            retrieved=[_doc("a", 0.9)],
            gold_ids=set(),
            reranked=None,
            answer=None,
            top_k=1,
        )
        sig = compute_signals(trace)
        assert sig.answer_present is False
        assert sig.answer_quality_ok is False

    def test_whitespace_only_answer_flags_invalid(self):
        """Whitespace-only answer -> answer_present=False, answer_quality_ok=False."""
        trace = QueryTrace(
            query="q",
            retrieved=[_doc("a", 0.9)],
            gold_ids=set(),
            reranked=None,
            answer="   \n\t  ",
            top_k=1,
        )
        sig = compute_signals(trace)
        assert sig.answer_present is False
        assert sig.answer_quality_ok is False

    def test_garbage_answer_flags_invalid(self):
        """Garbage token leak -> answer_quality_ok=False."""
        trace = QueryTrace(
            query="q",
            retrieved=[_doc("a", 0.9)],
            gold_ids=set(),
            reranked=None,
            answer="<|im_start|>Hello world this is a test sentence here.",
            top_k=1,
        )
        sig = compute_signals(trace)
        # present (non-empty) but not quality-ok
        assert sig.answer_present is True
        assert sig.answer_quality_ok is False

    def test_valid_answer_passes_quality(self):
        """A normal multi-sentence answer is quality-ok."""
        trace = QueryTrace(
            query="What is retrieval-augmented generation?",
            retrieved=[_doc("a", 0.9, text="RAG combines retrieval with generation.")],
            gold_ids=set(),
            reranked=None,
            answer=(
                "Retrieval-augmented generation (RAG) is a technique that combines "
                "information retrieval with language model generation. "
                "It first retrieves relevant documents, then uses them as context."
            ),
            top_k=1,
        )
        sig = compute_signals(trace)
        assert sig.answer_present is True
        assert sig.answer_quality_ok is True


# ---------------------------------------------------------------------------
# answer_grounded
# ---------------------------------------------------------------------------

class TestAnswerGrounded:
    def test_answer_grounded_high_overlap(self):
        """Answer shares many content words with retrieved text -> grounded=True."""
        retrieved_text = (
            "Qdrant is a vector similarity search engine. "
            "It provides a production-ready service with a convenient API."
        )
        answer = (
            "Qdrant is a vector similarity search engine that provides "
            "a convenient API for production use."
        )
        trace = QueryTrace(
            query="what is qdrant",
            retrieved=[_doc("a", 0.9, text=retrieved_text)],
            gold_ids=set(),
            reranked=None,
            answer=answer,
            top_k=1,
        )
        sig = compute_signals(trace)
        assert sig.answer_grounded is True

    def test_answer_grounded_low_overlap(self):
        """Answer shares few content words with retrieved text -> grounded=False."""
        retrieved_text = "The capital of France is Paris and it is a beautiful city."
        answer = (
            "Python was created by Guido van Rossum in 1991. "
            "It is a programming language with dynamic typing."
        )
        trace = QueryTrace(
            query="q",
            retrieved=[_doc("a", 0.9, text=retrieved_text)],
            gold_ids=set(),
            reranked=None,
            answer=answer,
            top_k=1,
        )
        sig = compute_signals(trace)
        assert sig.answer_grounded is False

    def test_answer_grounded_no_text_field_defaults_true(self):
        """If retrieved dicts have no 'text' key, grounded defaults to True (tolerate)."""
        trace = QueryTrace(
            query="q",
            retrieved=[_doc("a", 0.9)],  # no text field
            gold_ids=set(),
            reranked=None,
            answer="Some answer about some topic.",
            top_k=1,
        )
        sig = compute_signals(trace)
        assert sig.answer_grounded is True

    def test_no_answer_grounded_false(self):
        """No answer -> grounded=False."""
        trace = QueryTrace(
            query="q",
            retrieved=[_doc("a", 0.9, text="some text here")],
            gold_ids=set(),
            reranked=None,
            answer=None,
            top_k=1,
        )
        sig = compute_signals(trace)
        assert sig.answer_grounded is False


# ---------------------------------------------------------------------------
# n_retrieved
# ---------------------------------------------------------------------------

class TestNRetrieved:
    def test_n_retrieved_matches_list_length(self):
        docs = [_doc(f"d{i}", 0.9 - i * 0.1) for i in range(4)]
        trace = QueryTrace(
            query="q",
            retrieved=docs,
            gold_ids=set(),
            reranked=None,
            answer=None,
            top_k=4,
        )
        sig = compute_signals(trace)
        assert sig.n_retrieved == 4

    def test_n_retrieved_empty(self):
        trace = QueryTrace(
            query="q",
            retrieved=[],
            gold_ids=set(),
            reranked=None,
            answer=None,
            top_k=5,
        )
        sig = compute_signals(trace)
        assert sig.n_retrieved == 0


# ---------------------------------------------------------------------------
# Signals is a dataclass with expected fields
# ---------------------------------------------------------------------------

class TestSignalsContract:
    def test_signals_has_expected_fields(self):
        trace = QueryTrace(
            query="q",
            retrieved=[_doc("a", 0.9)],
            gold_ids=set(),
            reranked=None,
            answer=None,
            top_k=1,
        )
        sig = compute_signals(trace)
        # All fields must be accessible without AttributeError
        _ = sig.recall_hit
        _ = sig.gold_rank
        _ = sig.score_gap
        _ = sig.reranker_moved_gold_down
        _ = sig.answer_present
        _ = sig.answer_quality_ok
        _ = sig.answer_grounded
        _ = sig.n_retrieved
