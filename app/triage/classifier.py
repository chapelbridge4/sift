"""Rule-based RAG failure classifier for sift triage.

Maps the deterministic :class:`~app.triage.signals.Signals` of a single query
into one or more :class:`~app.triage.taxonomy.RAGFailureType` verdicts with a
confidence and a per-stage attribution.

Design
------
* Deterministic rules run in priority order: a retrieval miss is diagnosed
  before a reranking demotion, which is diagnosed before generation-stage
  problems. The earliest stage that breaks is the root cause worth reporting.
* The optional local LLM judge (disabled by default via
  ``settings.TRIAGE_USE_LLM_JUDGE``) only disambiguates generation-stage
  failures and is imported lazily, so ``classify`` stays import-side-effect-free
  and never loads a model unless explicitly asked.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.triage.signals import QueryTrace, compute_signals
from app.triage.taxonomy import RAGFailureType

# Confidence levels per rule (higher = less ambiguous signal).
_CONF_RETRIEVAL_MISS = 0.9
_CONF_RERANK_DEMOTED = 0.85
_CONF_GENERATION_QUALITY = 0.7
_CONF_GENERATION_GROUNDING = 0.6


@dataclass
class TriageVerdict:
    """The triage outcome for one query.

    ``failure_types`` is ordered by confidence (highest first). A passing query
    yields an empty ``failure_types`` and ``primary_stage`` of ``None``.
    """

    failure_types: list[tuple[RAGFailureType, float]]
    primary_stage: str | None
    evidence: str


def _resolve_use_judge(use_llm_judge: bool | None) -> bool:
    if use_llm_judge is not None:
        return use_llm_judge
    try:
        from app.config import get_settings

        return bool(getattr(get_settings(), "TRIAGE_USE_LLM_JUDGE", False))
    except Exception:
        return False


def classify(trace: QueryTrace, *, use_llm_judge: bool | None = None) -> TriageVerdict:
    """Classify why a RAG query failed (or confirm it passed).

    Rules, in priority order:
    1. gold never retrieved              -> RELEVANT_NOT_RETRIEVED (retrieval)
    2. gold retrieved but reranked down  -> RELEVANT_DEMOTED       (reranking)
    3. no/garbage answer                 -> INCOMPLETE/UNFAITHFUL  (generation)
    4. valid answer ungrounded in context-> CONTEXT_IGNORED        (generation)
    Otherwise the query passed.
    """
    s = compute_signals(trace)

    # 1. Retrieval miss — the relevant document never made it into the candidates.
    if not s.recall_hit:
        ft = RAGFailureType.RELEVANT_NOT_RETRIEVED
        return TriageVerdict(
            failure_types=[(ft, _CONF_RETRIEVAL_MISS)],
            primary_stage=ft.stage,
            evidence="recall_hit is False: no gold document was retrieved in the top-k candidates.",
        )

    # 2. Reranking demotion — retrieval found the gold doc, the reranker buried it.
    if s.reranker_moved_gold_down:
        ft = RAGFailureType.RELEVANT_DEMOTED
        return TriageVerdict(
            failure_types=[(ft, _CONF_RERANK_DEMOTED)],
            primary_stage=ft.stage,
            evidence="reranker_moved_gold_down is True: the reranker demoted the gold document below its retrieval rank.",
        )

    # 3. Generation — no answer was produced despite good retrieval.
    if not s.answer_present:
        ft = RAGFailureType.INCOMPLETE
        verdict = TriageVerdict(
            failure_types=[(ft, _CONF_GENERATION_QUALITY)],
            primary_stage=ft.stage,
            evidence="good retrieval but no answer was produced (answer_present=False): the generation step returned nothing usable.",
        )
        return _maybe_refine_generation(trace, verdict, use_llm_judge)

    # 4. Generation — an answer is present but ignores the retrieved context.
    #    Deterministic groundedness is a lexical proxy. We do NOT gate on the
    #    bundled answer_quality_ok signal here: it relies on a >=2-sentence
    #    coherence rule that false-positives on valid short answers. True
    #    UNFAITHFUL hallucination that *does* overlap the context is left to the
    #    optional local LLM judge (see _maybe_refine_generation).
    if not s.answer_grounded:
        ft = RAGFailureType.CONTEXT_IGNORED
        verdict = TriageVerdict(
            failure_types=[(ft, _CONF_GENERATION_GROUNDING)],
            primary_stage=ft.stage,
            evidence="answer is present but not grounded in the retrieved context (low lexical overlap): the model ignored the provided context.",
        )
        return _maybe_refine_generation(trace, verdict, use_llm_judge)

    # Passed: gold retrieved, not demoted, valid grounded answer.
    return TriageVerdict(
        failure_types=[],
        primary_stage=None,
        evidence="no failure detected: gold retrieved, not demoted, answer present, and grounded in context.",
    )


def _maybe_refine_generation(
    trace: QueryTrace, verdict: TriageVerdict, use_llm_judge: bool | None
) -> TriageVerdict:
    """Optionally use a local LLM judge to disambiguate generation failures.

    Disabled by default. Never loads a model unless ``use_llm_judge`` resolves
    True. Any failure falls back to the deterministic ``verdict``.
    """
    if not _resolve_use_judge(use_llm_judge):
        return verdict
    try:  # pragma: no cover - exercised only when a local model is configured
        from app.services.inference import get_inference_backend

        backend = get_inference_backend()
        contexts = [d.get("text", "") for d in trace.retrieved if d.get("text")]
        prompt = (
            "Classify the generation failure as exactly one of "
            "UNFAITHFUL, INCOMPLETE, or CONTEXT_IGNORED.\n"
            f"Question: {trace.query}\nAnswer: {trace.answer}\n"
            f"Context: {' '.join(contexts)[:1500]}\nLabel:"
        )
        import asyncio

        label = asyncio.run(
            backend.generate_rag_response(query=prompt, retrieved_contexts=contexts, max_tokens=8)
        ).strip().upper()
        for ft in (RAGFailureType.UNFAITHFUL, RAGFailureType.INCOMPLETE, RAGFailureType.CONTEXT_IGNORED):
            if ft.name in label:
                return TriageVerdict(
                    failure_types=[(ft, 0.75)],
                    primary_stage=ft.stage,
                    evidence=verdict.evidence + f" | local LLM judge -> {ft.name}",
                )
        return verdict
    except Exception:
        return verdict
