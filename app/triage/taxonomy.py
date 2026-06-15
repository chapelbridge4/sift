"""RAG failure taxonomy for per-query triage.

Defines a structured enumeration of 16 failure modes that can occur in a
retrieval-augmented generation pipeline, organised across four pipeline
stages: chunking, retrieval, reranking, and generation.

Inspiration
-----------
The stage-based decomposition of RAG errors is informed by the framing in:

    Siriwardhana et al. (2023) and the layer6ai EACL-2026 RAG-error-
    classification work, which categorise RAG failures by pipeline stage
    and propose structured remediation strategies.

This module is clean-room and original: no code, taxonomy text, or
enumeration values were copied from any external source. All member names,
descriptions, and fix_hint strings were written from first principles for
the sift project.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class _FailureMeta(NamedTuple):
    """Immutable metadata attached to each RAGFailureType member."""

    stage: str
    description: str
    fix_hint: str


# ---------------------------------------------------------------------------
# Ordered pipeline stages
# ---------------------------------------------------------------------------

STAGES: list[str] = ["chunking", "retrieval", "reranking", "generation"]


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------

class RAGFailureType(Enum):
    """Enumeration of the 16 canonical RAG failure modes across 4 stages.

    Each member's value is a :class:`_FailureMeta` namedtuple exposing
    ``stage``, ``description``, and ``fix_hint`` as typed attributes.

    Usage
    -----
    >>> RAGFailureType.UNFAITHFUL.stage
    'generation'
    >>> RAGFailureType.UNFAITHFUL.description
    'Generated answer contains claims not supported by retrieved context.'
    """

    # ------------------------------------------------------------------
    # Stage: chunking
    # ------------------------------------------------------------------

    CHUNK_TOO_LARGE = _FailureMeta(
        stage="chunking",
        description=(
            "Chunks are so large that irrelevant text dilutes the relevant"
            " passage, harming both retrieval precision and generation focus."
        ),
        fix_hint=(
            "Reduce chunk_size; consider token-based splitting and verify"
            " that key facts fit inside a single, smaller chunk."
        ),
    )

    CHUNK_TOO_SMALL = _FailureMeta(
        stage="chunking",
        description=(
            "Chunks are so small that they lack the surrounding context"
            " needed for the model to interpret a retrieved passage correctly."
        ),
        fix_hint=(
            "Increase chunk_size or add a sentence-overlap window so that"
            " each chunk carries enough context to stand alone."
        ),
    )

    SEMANTIC_SPLIT = _FailureMeta(
        stage="chunking",
        description=(
            "A fixed-size or character-based splitter cuts mid-sentence or"
            " mid-argument, breaking the semantic unit the query needs."
        ),
        fix_hint=(
            "Switch to a semantic or sentence-aware splitter; validate that"
            " no chunk ends inside a sentence using an NLP tokeniser."
        ),
    )

    CONTEXT_BOUNDARY_LOST = _FailureMeta(
        stage="chunking",
        description=(
            "Section headers, table captions, or list structures are split"
            " across chunk boundaries, stripping the context that gives the"
            " contained facts meaning."
        ),
        fix_hint=(
            "Use a structure-aware splitter that respects Markdown/HTML"
            " headings and list blocks, keeping structural parents with"
            " their children."
        ),
    )

    # ------------------------------------------------------------------
    # Stage: retrieval
    # ------------------------------------------------------------------

    RELEVANT_NOT_RETRIEVED = _FailureMeta(
        stage="retrieval",
        description=(
            "The chunk that contains the correct answer exists in the index"
            " but does not appear in the top-k results for the query."
        ),
        fix_hint=(
            "Increase top_k, try hybrid (dense + sparse) retrieval, or"
            " use query expansion / HyDE to bridge the vocabulary gap."
        ),
    )

    IRRELEVANT_RETRIEVED = _FailureMeta(
        stage="retrieval",
        description=(
            "Retrieved chunks are topically close but factually unrelated"
            " to the query, introducing noise the model may hallucinate from."
        ),
        fix_hint=(
            "Add a metadata filter (date, source, category) or tighten the"
            " similarity threshold to exclude borderline matches."
        ),
    )

    EMBEDDING_MISMATCH = _FailureMeta(
        stage="retrieval",
        description=(
            "Index embeddings and query embeddings were produced by different"
            " models or different normalisation settings, degrading cosine"
            " similarity scores globally."
        ),
        fix_hint=(
            "Re-embed the entire index with the same model and settings used"
            " at query time; store the embedding model ID alongside index"
            " metadata and assert equality on startup."
        ),
    )

    QUERY_INTENT_MISPARSE = _FailureMeta(
        stage="retrieval",
        description=(
            "The query embedding captures surface form rather than intent,"
            " so semantically equivalent paraphrases retrieve different chunks."
        ),
        fix_hint=(
            "Apply query rewriting or intent classification before embedding;"
            " evaluate retrieval consistency across paraphrase sets."
        ),
    )

    # ------------------------------------------------------------------
    # Stage: reranking
    # ------------------------------------------------------------------

    RELEVANT_DEMOTED = _FailureMeta(
        stage="reranking",
        description=(
            "A reranker moves the gold chunk below the context window cutoff,"
            " so the generator never sees it despite retrieval succeeding."
        ),
        fix_hint=(
            "Audit reranker scores on known-good pairs; consider a score"
            " floor that prevents any first-pass top result from falling"
            " below rank N."
        ),
    )

    IRRELEVANT_PROMOTED = _FailureMeta(
        stage="reranking",
        description=(
            "The reranker elevates a plausible-sounding but factually wrong"
            " chunk to the top position, displacing correct evidence."
        ),
        fix_hint=(
            "Fine-tune the reranker on domain-specific negative examples;"
            " add a factual-consistency check before accepting the final"
            " ranking."
        ),
    )

    RERANKER_NOOP = _FailureMeta(
        stage="reranking",
        description=(
            "The reranker produces rankings nearly identical to the initial"
            " retrieval order, providing no signal and wasting latency."
        ),
        fix_hint=(
            "Measure rank-correlation between retrieval and reranker outputs;"
            " if Kendall τ > 0.95 consistently, skip or replace the reranker."
        ),
    )

    DIVERSITY_COLLAPSE = _FailureMeta(
        stage="reranking",
        description=(
            "The reranker selects multiple near-duplicate chunks, filling the"
            " context window with redundant text at the expense of coverage."
        ),
        fix_hint=(
            "Apply Maximal Marginal Relevance (MMR) or a diversity penalty"
            " after scoring to ensure the final context set covers distinct"
            " aspects of the query."
        ),
    )

    # ------------------------------------------------------------------
    # Stage: generation
    # ------------------------------------------------------------------

    UNFAITHFUL = _FailureMeta(
        stage="generation",
        description=(
            "Generated answer contains claims not supported by retrieved"
            " context; the model hallucinated rather than grounded its output."
        ),
        fix_hint=(
            "Add a faithfulness check (NLI or entailment) post-generation;"
            " use a system prompt that instructs the model to cite chunk IDs"
            " for every factual claim."
        ),
    )

    INCOMPLETE = _FailureMeta(
        stage="generation",
        description=(
            "The answer is partially correct but omits information that is"
            " present in the retrieved context and necessary to fully answer"
            " the query."
        ),
        fix_hint=(
            "Increase max_new_tokens; check whether relevant context is being"
            " truncated before reaching the model; use a completeness prompt"
            " that enumerates required sub-questions."
        ),
    )

    CONTEXT_IGNORED = _FailureMeta(
        stage="generation",
        description=(
            "The model answers from parametric memory and ignores the provided"
            " context entirely, even when context contains the correct answer."
        ),
        fix_hint=(
            "Strengthen the system prompt to foreground context use; test"
            " with a model known to follow RAG-style instructions; verify"
            " context token position (early is often better attended to)."
        ),
    )

    FORMAT_ERROR = _FailureMeta(
        stage="generation",
        description=(
            "The model output does not conform to the requested structure"
            " (JSON, numbered list, specific schema), breaking downstream"
            " parsing even if the factual content is correct."
        ),
        fix_hint=(
            "Use structured output / constrained decoding if the model"
            " supports it; add a format validation step and re-prompt once"
            " on schema violation."
        ),
    )

    # ------------------------------------------------------------------
    # Metadata properties (delegate to the NamedTuple value)
    # ------------------------------------------------------------------

    @property
    def stage(self) -> str:
        """The pipeline stage at which this failure occurs."""
        return self.value.stage

    @property
    def description(self) -> str:
        """One-sentence description of the failure mode."""
        return self.value.description

    @property
    def fix_hint(self) -> str:
        """Concrete remediation hint for this failure type."""
        return self.value.fix_hint


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def by_stage() -> dict[str, list[RAGFailureType]]:
    """Return all failure types grouped by pipeline stage.

    Returns
    -------
    dict[str, list[RAGFailureType]]
        Keys are the four stage names from :data:`STAGES`; values are lists
        of :class:`RAGFailureType` members belonging to that stage.  Every
        member appears in exactly one list (partition), and all four stage
        keys are always present.

    Example
    -------
    >>> grouped = by_stage()
    >>> len(grouped["chunking"])
    4
    """
    result: dict[str, list[RAGFailureType]] = {stage: [] for stage in STAGES}
    for member in RAGFailureType:
        result[member.stage].append(member)
    return result
