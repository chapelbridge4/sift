"""
Focused test for RagOrchestrator startup boundary.

Verifies that initialize() does not eagerly load the LLM/model.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class TestOrchestratorStartupBoundary(unittest.TestCase):
    """Test that RagOrchestrator.initialize() is lazy."""

    @staticmethod
    def _make_orchestrator(mock_llm):
        # Inject the backend (DI) instead of patching a module-level class.
        from app.pipeline.orchestrator import RagOrchestrator
        return RagOrchestrator(llm_service=mock_llm)

    def test_initialize_does_not_call_llm_service_initialize(self):
        """
        Verify initialize() does not call llm_service.initialize().

        Regression test for the startup boundary: calling
        RagOrchestrator.initialize() during generic startup must NOT
        force model loading.
        """
        mock_llm = MagicMock()
        mock_llm.initialize = AsyncMock()
        mock_llm.health_check = AsyncMock(return_value=True)
        with patch("app.pipeline.orchestrator.DocumentStore") as MockDocumentStore, \
             patch("app.pipeline.orchestrator.Reranker"), \
             patch("app.pipeline.orchestrator.ConversationMemory"):
            MockDocumentStore.return_value.initialize = AsyncMock()
            pfc = self._make_orchestrator(mock_llm)
            asyncio.run(pfc.initialize())
            mock_llm.initialize.assert_not_called()

    def test_document_store_initialized_on_initialize(self):
        """Verify the document store is initialized when initialize() is called."""
        mock_llm = MagicMock()
        mock_llm.initialize = AsyncMock()
        with patch("app.pipeline.orchestrator.DocumentStore") as MockDocumentStore, \
             patch("app.pipeline.orchestrator.Reranker"), \
             patch("app.pipeline.orchestrator.ConversationMemory"):
            mock_doc_store = MockDocumentStore.return_value
            mock_doc_store.initialize = AsyncMock()
            pfc = self._make_orchestrator(mock_llm)
            asyncio.run(pfc.initialize())
            mock_doc_store.initialize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
