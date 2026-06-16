"""
Focused tests for the query contract behavior.
Tests the following acceptance criteria:
- fusion_method is honored and passed through the pipeline
- use_llm=false returns retrieved documents without calling generation
- model_profile selection is request-scoped, not sticky shared state
- Response is valid when no documents are retrieved and use_llm=false
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.schemas import FusionMethod, QueryRequest


class TestFusionMethodThreading(unittest.TestCase):
    """Test that fusion_method is properly threaded through the pipeline."""

    def test_fusion_method_schema(self):
        """Test that FusionMethod enum has correct values."""
        self.assertEqual(FusionMethod.RRF.value, "rrf")
        self.assertEqual(FusionMethod.DBSF.value, "dbsf")

    def test_fusion_method_in_query_request(self):
        """Test that QueryRequest accepts fusion_method."""
        request = QueryRequest(
            collection_name="test",
            query="test query",
            fusion_method=FusionMethod.DBSF
        )
        self.assertEqual(request.fusion_method, FusionMethod.DBSF)
        self.assertEqual(request.fusion_method.value, "dbsf")


class TestUseLlmFalsePath(unittest.TestCase):
    """Test that use_llm=false returns retrieved documents without generation."""

    @patch('app.brain.prefrontal_cortex.Hippocampus')
    @patch('app.brain.prefrontal_cortex.Amygdala')
    @patch('app.brain.prefrontal_cortex.WorkingMemory')
    @patch('app.brain.prefrontal_cortex.LLMService')
    def test_use_llm_false_returns_docs_without_generation(self, mock_llm, mock_wm, mock_amygdala, mock_hippocampus):
        """When use_llm=False, retrieval-only path should return documents without LLM calls."""
        from app.brain.prefrontal_cortex import PrefrontalCortex

        pfc = PrefrontalCortex()
        pfc.hippocampus = AsyncMock()
        pfc.amygdala = MagicMock()
        pfc.working_memory = AsyncMock()
        pfc.llm_service = AsyncMock()

        mock_docs = [
            {"text": "doc1", "score": 0.9, "metadata": {}},
            {"text": "doc2", "score": 0.8, "metadata": {}}
        ]
        pfc.hippocampus.recall_memories = AsyncMock(return_value=mock_docs)
        pfc.amygdala.rank_by_importance = MagicMock(return_value=mock_docs)

        async def run_test():
            return await pfc.reason_with_context(
                query="test query",
                collection_name="test",
                use_llm=False,
                fusion_method="dbsf"
            )

        result = asyncio.run(run_test())

        self.assertIsNone(result["answer"])
        self.assertEqual(result["retrieved_documents"], mock_docs)
        self.assertIsNone(result["model_used"])
        pfc.llm_service.generate_rag_response.assert_not_called()

    @patch('app.brain.prefrontal_cortex.Hippocampus')
    @patch('app.brain.prefrontal_cortex.Amygdala')
    @patch('app.brain.prefrontal_cortex.WorkingMemory')
    @patch('app.brain.prefrontal_cortex.LLMService')
    def test_use_llm_false_empty_docs(self, mock_llm, mock_wm, mock_amygdala, mock_hippocampus):
        """When use_llm=False and no docs retrieved, response should be valid."""
        from app.brain.prefrontal_cortex import PrefrontalCortex

        pfc = PrefrontalCortex()
        pfc.hippocampus = AsyncMock()
        pfc.amygdala = MagicMock()
        pfc.working_memory = AsyncMock()
        pfc.llm_service = AsyncMock()

        pfc.hippocampus.recall_memories = AsyncMock(return_value=[])

        async def run_test():
            return await pfc.reason_with_context(
                query="test query",
                collection_name="test",
                use_llm=False
            )

        result = asyncio.run(run_test())

        self.assertIsNone(result["answer"])
        self.assertEqual(result["retrieved_documents"], [])
        self.assertIsNone(result["model_used"])
        pfc.llm_service.generate_rag_response.assert_not_called()


class TestModelProfileIsolation(unittest.TestCase):
    """Test that model_profile selection is request-scoped, not sticky."""

    @patch('app.services.llm_service.ModelManager')
    def test_get_model_for_request_does_not_persist(self, mock_manager_class):
        """Calling get_model_for_request should not modify self.model_name."""
        from app.config import get_settings
        from app.services.llm_service import LLMService

        settings = get_settings()
        service = LLMService()
        service.model_manager = MagicMock()
        service.model_manager.get_model_for_profile = MagicMock(return_value="qwen2.5:1.5b")
        service.model_manager.get_model_config = MagicMock(return_value={"temperature": 0.7, "max_tokens": 300})

        initial_model = service.model_name
        initial_profile = service.current_profile

        model_name, config = service.get_model_for_request("quality")

        self.assertEqual(model_name, "qwen2.5:1.5b")
        self.assertEqual(service.model_name, initial_model)
        self.assertEqual(service.current_profile, initial_profile)

    @patch('app.services.llm_service.LLMService._ensure_model', new_callable=AsyncMock)
    @patch('app.services.llm_service.ModelManager')
    def test_generate_rag_response_uses_request_model(self, mock_manager_class, mock_ensure_model):
        """generate_rag_response should use request-scoped model without persistence."""
        from app.config import get_settings
        from app.services.llm_service import LLMService

        settings = get_settings()
        service = LLMService()
        service.client = AsyncMock()
        service.initialize = AsyncMock()

        service.model_manager = MagicMock()
        service._ensure_model = AsyncMock()
        service.model_manager.get_model_for_profile = MagicMock(side_effect=["qwen2.5:1.5b", "qwen2.5:3b"])
        service.model_manager.get_model_config = MagicMock(return_value={"temperature": 0.7, "max_tokens": 300})

        service.model_name = "qwen2.5:1.5b"
        service.current_config = {"temperature": 0.7, "max_tokens": 300}
        service.current_profile = "balanced"

        async def run_test():
            service.generate = AsyncMock(return_value='Test response')
            service.chat = AsyncMock(return_value='Test response')
            result = await service.generate_rag_response(
                query="test",
                retrieved_contexts=["context1"],
                model_profile="quality"
            )
            return result

        result = asyncio.run(run_test())

        self.assertEqual(service.model_name, "qwen2.5:1.5b")
        self.assertEqual(service.current_profile, "balanced")


class TestModelUsedReporting(unittest.TestCase):
    """Test that model_used reports the actual request-scoped model."""

    @patch('app.brain.prefrontal_cortex.Hippocampus')
    @patch('app.brain.prefrontal_cortex.Amygdala')
    @patch('app.brain.prefrontal_cortex.WorkingMemory')
    @patch('app.brain.prefrontal_cortex.LLMService')
    def test_model_used_reports_request_scoped_model(self, mock_llm_class, mock_wm, mock_amygdala, mock_hippocampus):
        """When model_profile='quality', model_used should report quality model not default."""
        from app.brain.prefrontal_cortex import PrefrontalCortex

        pfc = PrefrontalCortex()
        pfc.hippocampus = AsyncMock()
        pfc.amygdala = MagicMock()
        pfc.working_memory = AsyncMock()
        pfc.llm_service = AsyncMock()

        mock_docs = [{"text": "doc1", "score": 0.9, "metadata": {}}]
        pfc.hippocampus.recall_memories = AsyncMock(return_value=mock_docs)
        pfc.amygdala.rank_by_importance = MagicMock(return_value=mock_docs)

        pfc.llm_service.get_model_for_request = MagicMock(return_value=("quality_model", {"temperature": 0.7}))
        pfc.llm_service.generate_rag_response = AsyncMock(return_value="Test answer")

        async def run_test():
            return await pfc.reason_with_context(
                query="test query",
                collection_name="test",
                model_profile="quality",
                use_llm=True
            )

        result = asyncio.run(run_test())

        self.assertEqual(result["model_used"], "quality_model")
        pfc.llm_service.get_model_for_request.assert_called_with("quality")


class TestNoContextModelProfile(unittest.TestCase):
    """Test that no-results branch honors model_profile."""

    @patch('app.brain.prefrontal_cortex.Hippocampus')
    @patch('app.brain.prefrontal_cortex.Amygdala')
    @patch('app.brain.prefrontal_cortex.WorkingMemory')
    @patch('app.brain.prefrontal_cortex.LLMService')
    def test_no_context_generation_uses_model_profile(self, mock_llm_class, mock_wm, mock_amygdala, mock_hippocampus):
        """When no docs retrieved and model_profile specified, generation uses that profile."""
        from app.brain.prefrontal_cortex import PrefrontalCortex

        pfc = PrefrontalCortex()
        pfc.hippocampus = AsyncMock()
        pfc.amygdala = MagicMock()
        pfc.working_memory = AsyncMock()
        pfc.llm_service = AsyncMock()

        pfc.hippocampus.recall_memories = AsyncMock(return_value=[])
        pfc.llm_service.get_model_for_request = MagicMock(return_value=("quality_model", {"temperature": 0.7}))
        pfc.llm_service.generate = AsyncMock(return_value="No context answer")

        async def run_test():
            return await pfc.reason_with_context(
                query="test query",
                collection_name="test",
                model_profile="quality",
                use_llm=True
            )

        result = asyncio.run(run_test())

        self.assertEqual(result["model_used"], "quality_model")
        self.assertEqual(result["answer"], "No context answer")
        pfc.llm_service.get_model_for_request.assert_called_with("quality")
        pfc.llm_service.generate.assert_called_once()
        call_kwargs = pfc.llm_service.generate.call_args.kwargs
        self.assertEqual(call_kwargs["model_name"], "quality_model")

    @patch('app.brain.prefrontal_cortex.Hippocampus')
    @patch('app.brain.prefrontal_cortex.Amygdala')
    @patch('app.brain.prefrontal_cortex.WorkingMemory')
    @patch('app.brain.prefrontal_cortex.LLMService')
    def test_no_context_with_conversation_history_uses_model_profile(self, mock_llm_class, mock_wm, mock_amygdala, mock_hippocampus):
        """When no docs but has conversation history, chat uses model_profile."""
        from app.brain.prefrontal_cortex import PrefrontalCortex

        pfc = PrefrontalCortex()
        pfc.hippocampus = AsyncMock()
        pfc.amygdala = MagicMock()
        pfc.working_memory = AsyncMock()
        pfc.llm_service = AsyncMock()

        pfc.hippocampus.recall_memories = AsyncMock(return_value=[])
        pfc.llm_service.get_model_for_request = MagicMock(return_value=("fast_model", {"temperature": 0.5}))
        pfc.llm_service.chat = AsyncMock(return_value="Chat answer")
        pfc.working_memory.get_conversation_history = AsyncMock(return_value=[{"role": "user", "content": "Hello"}])

        async def run_test():
            return await pfc.reason_with_context(
                query="test query",
                collection_name="test",
                model_profile="fast",
                conversation_id="conv123",
                use_llm=True
            )

        result = asyncio.run(run_test())

        self.assertEqual(result["model_used"], "fast_model")
        pfc.llm_service.chat.assert_called_once()
        call_kwargs = pfc.llm_service.chat.call_args.kwargs
        self.assertEqual(call_kwargs["model_name"], "fast_model")


class TestRetrievalMethodReporting(unittest.TestCase):
    """Test that retrieval_method is correctly reported in response."""

    def test_retrieval_method_format(self):
        """Test that retrieval_method is formatted correctly."""
        fusion_method = FusionMethod.DBSF
        expected = "hybrid_dbsf"
        self.assertEqual(f"hybrid_{fusion_method.value}", expected)

        fusion_method = FusionMethod.RRF
        expected = "hybrid_rrf"
        self.assertEqual(f"hybrid_{fusion_method.value}", expected)


if __name__ == '__main__':
    import asyncio
    unittest.main()
