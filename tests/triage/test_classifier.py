"""Tests for app/triage/classifier.py — rule-based RAG failure classifier.

Written TDD-first: all tests FAIL before classifier.py exists, then PASS
after implementation.  No model loading, no network, no I/O.
TRIAGE_USE_LLM_JUDGE is False/default throughout.
"""

from __future__ import annotations

from app.triage.signals import QueryTrace
from app.triage.taxonomy import RAGFailureType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(doc_id: str, score: float, text: str = "") -> dict:
    d = {"doc_id": doc_id, "score": score}
    if text:
        d["text"] = text
    return d


def _gold_not_retrieved() -> QueryTrace:
    """Gold doc absent from retrieved list → RELEVANT_NOT_RETRIEVED."""
    return QueryTrace(
        query="what is qdrant",
        retrieved=[_doc("a", 0.9), _doc("b", 0.8), _doc("c", 0.7)],
        gold_ids={"gold_doc"},
        reranked=None,
        answer="Qdrant is a vector database.",
        top_k=3,
    )


def _reranker_demoted() -> QueryTrace:
    """Gold retrieved at rank 1 but reranker pushes it down → RELEVANT_DEMOTED."""
    return QueryTrace(
        query="what is hybrid retrieval",
        retrieved=[_doc("gold", 0.95), _doc("other", 0.80)],
        gold_ids={"gold"},
        reranked=["other", "gold"],   # gold moved from rank 1 to rank 2
        answer="Hybrid retrieval combines dense and sparse signals.",
        top_k=2,
    )


def _good_retrieval_bad_answer() -> QueryTrace:
    """Gold retrieved and not demoted but answer is empty → UNFAITHFUL or INCOMPLETE."""
    return QueryTrace(
        query="explain attention mechanism",
        retrieved=[_doc("gold", 0.92, text="Attention mechanisms allow models to focus on relevant tokens.")],
        gold_ids={"gold"},
        reranked=None,
        answer="",          # empty → answer_quality_ok False
        top_k=1,
    )


def _good_retrieval_ungrounded_answer() -> QueryTrace:
    """Gold retrieved, valid answer text, but answer not grounded in retrieved text → CONTEXT_IGNORED."""
    return QueryTrace(
        query="what are transformers",
        retrieved=[_doc("gold", 0.90, text="transformers revolutionized natural language processing tasks")],
        gold_ids={"gold"},
        reranked=None,
        # Answer has NO content-word overlap with the retrieved text
        answer="Cats sleep frequently during daylight hours due to carnivore biology.",
        top_k=1,
    )


def _passing_query() -> QueryTrace:
    """Gold retrieved, good grounded answer → no failure."""
    retrieved_text = "transformers revolutionized natural language processing tasks sequence models"
    return QueryTrace(
        query="what are transformers",
        retrieved=[_doc("gold", 0.92, text=retrieved_text)],
        gold_ids={"gold"},
        reranked=None,
        answer="Transformers revolutionized natural language processing tasks.",
        top_k=1,
    )


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

from app.triage.classifier import TriageVerdict, classify  # noqa: E402

# ---------------------------------------------------------------------------
# TriageVerdict dataclass contract
# ---------------------------------------------------------------------------

class TestTriageVerdictContract:
    def test_has_required_fields(self):
        v = TriageVerdict(failure_types=[], primary_stage=None, evidence="ok")
        assert v.failure_types == []
        assert v.primary_stage is None
        assert v.evidence == "ok"

    def test_failure_types_are_tuples_of_type_and_confidence(self):
        v = TriageVerdict(
            failure_types=[(RAGFailureType.RELEVANT_NOT_RETRIEVED, 0.9)],
            primary_stage="retrieval",
            evidence="recall_hit is False",
        )
        ft, conf = v.failure_types[0]
        assert ft is RAGFailureType.RELEVANT_NOT_RETRIEVED
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Retrieval miss → RELEVANT_NOT_RETRIEVED
# ---------------------------------------------------------------------------

class TestRetrievalMiss:
    def test_failure_type_is_relevant_not_retrieved(self):
        verdict = classify(_gold_not_retrieved())
        types = [ft for ft, _ in verdict.failure_types]
        assert RAGFailureType.RELEVANT_NOT_RETRIEVED in types

    def test_primary_stage_is_retrieval(self):
        verdict = classify(_gold_not_retrieved())
        assert verdict.primary_stage == "retrieval"

    def test_confidence_is_high(self):
        verdict = classify(_gold_not_retrieved())
        conf = next(c for ft, c in verdict.failure_types if ft is RAGFailureType.RELEVANT_NOT_RETRIEVED)
        assert conf >= 0.85, f"Expected high confidence for retrieval miss, got {conf}"

    def test_evidence_mentions_recall(self):
        verdict = classify(_gold_not_retrieved())
        assert "recall" in verdict.evidence.lower() or "retrieved" in verdict.evidence.lower()


# ---------------------------------------------------------------------------
# Reranker demotion → RELEVANT_DEMOTED
# ---------------------------------------------------------------------------

class TestRerankerDemotion:
    def test_failure_type_is_relevant_demoted(self):
        verdict = classify(_reranker_demoted())
        types = [ft for ft, _ in verdict.failure_types]
        assert RAGFailureType.RELEVANT_DEMOTED in types

    def test_primary_stage_is_reranking(self):
        verdict = classify(_reranker_demoted())
        assert verdict.primary_stage == "reranking"

    def test_evidence_mentions_rerank(self):
        verdict = classify(_reranker_demoted())
        assert "rerank" in verdict.evidence.lower() or "demot" in verdict.evidence.lower()


# ---------------------------------------------------------------------------
# Good retrieval but bad answer (empty) → generation failure
# ---------------------------------------------------------------------------

class TestGenerationQualityFailure:
    def test_failure_type_is_generation_stage(self):
        verdict = classify(_good_retrieval_bad_answer())
        types = [ft for ft, _ in verdict.failure_types]
        generation_types = {
            RAGFailureType.UNFAITHFUL,
            RAGFailureType.INCOMPLETE,
            RAGFailureType.CONTEXT_IGNORED,
        }
        assert any(t in generation_types for t in types), (
            f"Expected a generation-stage failure, got {types}"
        )

    def test_primary_stage_is_generation(self):
        verdict = classify(_good_retrieval_bad_answer())
        assert verdict.primary_stage == "generation"

    def test_empty_answer_yields_incomplete_or_unfaithful(self):
        verdict = classify(_good_retrieval_bad_answer())
        types = [ft for ft, _ in verdict.failure_types]
        assert (
            RAGFailureType.INCOMPLETE in types or RAGFailureType.UNFAITHFUL in types
        )


# ---------------------------------------------------------------------------
# Good retrieval, valid answer, but ungrounded → CONTEXT_IGNORED
# ---------------------------------------------------------------------------

class TestContextIgnored:
    def test_failure_type_is_context_ignored(self):
        verdict = classify(_good_retrieval_ungrounded_answer())
        types = [ft for ft, _ in verdict.failure_types]
        assert RAGFailureType.CONTEXT_IGNORED in types

    def test_primary_stage_is_generation(self):
        verdict = classify(_good_retrieval_ungrounded_answer())
        assert verdict.primary_stage == "generation"

    def test_evidence_mentions_grounding(self):
        verdict = classify(_good_retrieval_ungrounded_answer())
        assert "ground" in verdict.evidence.lower() or "context" in verdict.evidence.lower()


# ---------------------------------------------------------------------------
# Passing query → no failure
# ---------------------------------------------------------------------------

class TestPassingQuery:
    def test_no_failure_types(self):
        verdict = classify(_passing_query())
        assert verdict.failure_types == []

    def test_primary_stage_is_none(self):
        verdict = classify(_passing_query())
        assert verdict.primary_stage is None

    def test_evidence_indicates_pass(self):
        verdict = classify(_passing_query())
        assert "no failure" in verdict.evidence.lower()


# ---------------------------------------------------------------------------
# use_llm_judge=False must never load a model
# ---------------------------------------------------------------------------

class TestNoModelLoad:
    def test_classify_with_judge_false_does_not_import_inference(self):
        """classify(use_llm_judge=False) must not import app.services.inference."""
        import sys
        # Ensure inference module is NOT loaded as a side-effect
        before = set(sys.modules.keys())
        classify(_gold_not_retrieved(), use_llm_judge=False)
        after = set(sys.modules.keys())
        new_modules = after - before
        inference_loaded = any("inference" in m for m in new_modules)
        assert not inference_loaded, f"Inference module was loaded: {new_modules}"
