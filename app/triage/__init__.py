"""app.triage — RAG failure triage package.

Public API re-exported here for convenience.
"""

from app.triage.taxonomy import STAGES, RAGFailureType, by_stage

__all__ = ["RAGFailureType", "STAGES", "by_stage"]
