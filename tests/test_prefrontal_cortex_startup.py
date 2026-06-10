"""
Focused test for PrefrontalCortex startup boundary.

Verifies that initialize() does not eagerly load the LLM/model.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch


class TestPrefrontalCortexStartupBoundary(unittest.TestCase):
    """Test that PrefrontalCortex.initialize() is lazy."""

    def test_initialize_does_not_call_llm_service_initialize(self):
        """
        Verify initialize() does not call llm_service.initialize().

        This is a regression test for the startup boundary: calling
        PrefrontalCortex.initialize() during generic startup should NOT
        force model loading.
        """
        with patch("app.brain.prefrontal_cortex.LLMService") as MockLLMService, \
             patch("app.brain.prefrontal_cortex.Hippocampus") as MockHippocampus, \
             patch("app.brain.prefrontal_cortex.Amygdala") as MockAmygdala, \
             patch("app.brain.prefrontal_cortex.WorkingMemory") as MockWorkingMemory:

            mock_llm_instance = MockLLMService.return_value
            mock_llm_instance.initialize = AsyncMock()
            mock_llm_instance.health_check = AsyncMock(return_value=True)

            mock_hippo_instance = MockHippocampus.return_value
            mock_hippo_instance.initialize = AsyncMock()

            from app.brain.prefrontal_cortex import PrefrontalCortex

            pfc = PrefrontalCortex()
            asyncio.run(pfc.initialize())

            mock_llm_instance.initialize.assert_not_called()

    def test_hippocampus_initialized_on_initialize(self):
        """Verify hippocampus is initialized when initialize() is called."""
        with patch("app.brain.prefrontal_cortex.LLMService") as MockLLMService, \
             patch("app.brain.prefrontal_cortex.Hippocampus") as MockHippocampus, \
             patch("app.brain.prefrontal_cortex.Amygdala") as MockAmygdala, \
             patch("app.brain.prefrontal_cortex.WorkingMemory") as MockWorkingMemory:

            mock_llm_instance = MockLLMService.return_value
            mock_llm_instance.initialize = AsyncMock()

            mock_hippo_instance = MockHippocampus.return_value
            mock_hippo_instance.initialize = AsyncMock()

            from app.brain.prefrontal_cortex import PrefrontalCortex

            pfc = PrefrontalCortex()
            asyncio.run(pfc.initialize())

            mock_hippo_instance.initialize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
