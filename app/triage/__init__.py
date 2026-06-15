"""app.triage — RAG failure triage package.

Public API re-exported here for convenience.
"""

from app.triage.taxonomy import RAGFailureType, STAGES, by_stage

__all__ = ["RAGFailureType", "STAGES", "by_stage"]
