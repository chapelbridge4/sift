"""Deterministic per-query span signals for RAG failure triage.

All functions are pure and deterministic.  No model loading, no I/O, no
network calls.  The classifier (next task) consumes :class:`Signals` directly.

Design notes
------------
* :func:`compute_signals` is the single public entry point.
* Quality helpers are imported lazily from ``app.tuning.quality`` to avoid
  circular imports and to keep the import side-effect-free.
* ``answer_grounded`` uses a cheap lexical proxy: the fraction of
  content-word tokens in the answer that appear in the concatenated retrieved
  texts.  Threshold 0.3 → True.  When retrieved dicts carry no ``"text"``
  key the signal defaults to ``True`` (tolerant) so the function stays safe
  for eval runners that supply only scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class QueryTrace:
    """All observable data for a single RAG pipeline execution."""

    query: str
    """The user query text."""

    retrieved: list[dict]
    """Ordered list of retrieved documents.

    Each entry must have at minimum ``{"doc_id": str, "score": float}``.
    An optional ``"text"`` key carries the chunk text used for groundedness.
    """

    gold_ids: set[str]
    """Known-correct document IDs.  May be empty when ground truth is unknown."""

    reranked: list[str] | None
    """Post-rerank doc_id order, or ``None`` when no reranker ran."""

    answer: str | None
    """Generated answer string, or ``None`` when generation was skipped."""

    top_k: int
    """Number of documents requested from retrieval."""


@dataclass
class Signals:
    """Deterministic boolean/numeric signals extracted from a :class:`QueryTrace`."""

    recall_hit: bool
    """True iff at least one gold doc appears in the retrieved list."""

    gold_rank: int | None
    """1-based best gold position in retrieved (smallest rank wins), else None."""

    score_gap: float
    """Top-1 score minus last retrieved score.  0.0 if fewer than 2 results."""

    reranker_moved_gold_down: bool
    """True when a reranker degraded the best gold rank compared to retrieval."""

    answer_present: bool
    """True iff ``answer`` is non-None and contains non-whitespace characters."""

    answer_quality_ok: bool
    """True iff quality helpers from ``app.tuning.quality`` accept the answer."""

    answer_grounded: bool
    """True iff answer content-words overlap retrieved texts above threshold."""

    n_retrieved: int
    """Number of documents in the retrieved list."""


# ---------------------------------------------------------------------------
# Stop-words for lexical grounding (English, minimal set)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "it", "its", "this", "that", "these",
    "those", "as", "if", "so", "not", "no", "nor", "yet", "both", "either",
    "than", "then", "when", "where", "which", "who", "whom", "what", "how",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> list[str]:
    """Lower-case alphabetic tokens with stop-words removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_WORDS]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _recall_and_rank(
    retrieved: list[dict],
    gold_ids: set[str],
) -> tuple[bool, int | None]:
    """Compute recall_hit and best gold_rank (1-based) from retrieved list."""
    if not gold_ids:
        return False, None

    best_rank: int | None = None
    for idx, doc in enumerate(retrieved):
        if doc.get("doc_id") in gold_ids:
            rank = idx + 1  # 1-based
            if best_rank is None or rank < best_rank:
                best_rank = rank

    return best_rank is not None, best_rank


def _score_gap(retrieved: list[dict]) -> float:
    """Top-1 score minus last retrieved score, or 0.0 when < 2 results."""
    if len(retrieved) < 2:
        return 0.0
    return float(retrieved[0].get("score", 0.0)) - float(retrieved[-1].get("score", 0.0))


def _reranker_moved_gold_down(
    retrieved: list[dict],
    reranked: list[str] | None,
    gold_ids: set[str],
) -> bool:
    """True iff best gold rank worsened after reranking.

    Returns False when reranked is None, gold_ids is empty, or gold is absent
    from both orderings.
    """
    if reranked is None or not gold_ids:
        return False

    # Best gold rank in pre-rerank order (1-based)
    pre_rank: int | None = None
    for idx, doc in enumerate(retrieved):
        if doc.get("doc_id") in gold_ids:
            rank = idx + 1
            if pre_rank is None or rank < pre_rank:
                pre_rank = rank

    if pre_rank is None:
        return False  # gold not in retrieved at all

    # Best gold rank in post-rerank order (1-based)
    post_rank: int | None = None
    for idx, doc_id in enumerate(reranked):
        if doc_id in gold_ids:
            rank = idx + 1
            if post_rank is None or rank < post_rank:
                post_rank = rank

    if post_rank is None:
        return False  # gold not in reranked list

    return post_rank > pre_rank


def _answer_quality(answer: str | None) -> tuple[bool, bool]:
    """Return (answer_present, answer_quality_ok).

    Uses ``detect_garbage`` and ``is_valid_response`` from
    ``app.tuning.quality`` (lazy import to avoid circular dependencies).
    Falls back gracefully when the import fails.
    """
    present = bool(answer and answer.strip())
    if not present:
        return False, False

    try:
        from app.tuning.quality import detect_garbage, is_valid_response  # noqa: PLC0415
        valid, _reason = is_valid_response(answer)
        quality_ok = valid and not detect_garbage(answer)
    except ImportError:
        # Tolerate missing optional dependency; assume quality is ok when we
        # cannot check.
        quality_ok = True

    return True, quality_ok


def _answer_grounded(
    answer: str | None,
    retrieved: list[dict],
    threshold: float = 0.3,
) -> bool:
    """Lexical groundedness proxy.

    Computes the fraction of content-word tokens in ``answer`` that appear
    in the concatenated retrieved texts.  Returns True when the fraction
    exceeds ``threshold``.

    Tolerant behaviour
    ------------------
    * No answer → False.
    * Retrieved dicts lack ``"text"`` key → True (can't evaluate, be lenient).
    * Answer has no content tokens → False.
    """
    if not answer or not answer.strip():
        return False

    # Collect retrieved texts
    texts = [doc["text"] for doc in retrieved if "text" in doc]

    if not texts:
        # No text fields available — tolerate and assume grounded
        return True

    corpus = " ".join(texts)
    corpus_tokens: set[str] = set(_content_tokens(corpus))

    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return False

    overlap = sum(1 for t in answer_tokens if t in corpus_tokens)
    fraction = overlap / len(answer_tokens)
    return fraction >= threshold


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_signals(trace: QueryTrace) -> Signals:
    """Compute all deterministic signals from a :class:`QueryTrace`.

    Parameters
    ----------
    trace:
        Fully populated trace for a single RAG pipeline execution.

    Returns
    -------
    Signals
        Struct of deterministic boolean/numeric features ready for the
        classifier layer.
    """
    recall_hit, gold_rank = _recall_and_rank(trace.retrieved, trace.gold_ids)
    score_gap = _score_gap(trace.retrieved)
    moved_down = _reranker_moved_gold_down(trace.retrieved, trace.reranked, trace.gold_ids)
    answer_present, answer_quality_ok = _answer_quality(trace.answer)
    grounded = _answer_grounded(trace.answer, trace.retrieved)

    return Signals(
        recall_hit=recall_hit,
        gold_rank=gold_rank,
        score_gap=score_gap,
        reranker_moved_gold_down=moved_down,
        answer_present=answer_present,
        answer_quality_ok=answer_quality_ok,
        answer_grounded=grounded,
        n_retrieved=len(trace.retrieved),
    )
