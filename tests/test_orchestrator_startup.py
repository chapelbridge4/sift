"""
Focused test for RagOrchestrator startup boundary.

Verifies that initialize() does not eagerly load the LLM/model.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch


class TestOrchestratorStartupBoundary(unittest.TestCase):
    """Test that RagOrchestrator.initialize() is lazy."""

    def test_initialize_does_not_call_llm_service_initialize(self):
        """
        Verify initialize() does not call llm_service.initialize().

        This is a regression test for the startup boundary: calling
        RagOrchestrator.initialize() during generic startup should NOT
        force model loading.
        """
        with patch("app.pipeline.orchestrator.LLMService") as MockLLMService, \
             patch("app.pipeline.orchestrator.DocumentStore") as MockDocumentStore, \
             patch("app.pipeline.orchestrator.Reranker") as MockReranker, \
             patch("app.pipeline.orchestrator.ConversationMemory") as MockConversationMemory:

            mock_llm_instance = MockLLMService.return_value
            mock_llm_instance.initialize = AsyncMock()
            mock_llm_instance.health_check = AsyncMock(return_value=True)

            mock_hippo_instance = MockDocumentStore.return_value
            mock_hippo_instance.initialize = AsyncMock()

            from app.pipeline.orchestrator import RagOrchestrator

            pfc = RagOrchestrator()
            asyncio.run(pfc.initialize())

            mock_llm_instance.initialize.assert_not_called()

    def test_document_store_initialized_on_initialize(self):
        """Verify the document store is initialized when initialize() is called."""
        with patch("app.pipeline.orchestrator.LLMService") as MockLLMService, \
             patch("app.pipeline.orchestrator.DocumentStore") as MockDocumentStore, \
             patch("app.pipeline.orchestrator.Reranker") as MockReranker, \
             patch("app.pipeline.orchestrator.ConversationMemory") as MockConversationMemory:

            mock_llm_instance = MockLLMService.return_value
            mock_llm_instance.initialize = AsyncMock()

            mock_hippo_instance = MockDocumentStore.return_value
            mock_hippo_instance.initialize = AsyncMock()

            from app.pipeline.orchestrator import RagOrchestrator

            pfc = RagOrchestrator()
            asyncio.run(pfc.initialize())

            mock_hippo_instance.initialize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
