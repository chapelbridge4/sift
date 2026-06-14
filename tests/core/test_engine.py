import importlib.util
import pytest
from unittest.mock import MagicMock, patch

# mlx_vlm is an Apple-Silicon-only optional dependency; skip MLX-specific tests
# when it isn't installed (e.g. Linux CI running the default GGUF backend).
_HAS_MLX_VLM = importlib.util.find_spec("mlx_vlm") is not None


class TestQwenEngine:
    @pytest.mark.skipif(not _HAS_MLX_VLM, reason="requires mlx_vlm (Apple Silicon optional dep)")
    @patch("mlx_vlm.load")
    def test_engine_loads_model_and_processor(self, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock())
        from app.core.engine import QwenEngine
        e = QwenEngine("mlx-community/Qwen3.5-4B-MLX-4bit")
        assert e.model_id == "mlx-community/Qwen3.5-4B-MLX-4bit"

    def test_build_prompt_prepends_no_think_for_qwen(self):
        pytest.skip("implementation detail, tested via integration")

    def test_sanitize_strips_thinking_blocks(self):
        from app.core.engine import QwenEngine
        engine = QwenEngine.__new__(QwenEngine)
        engine.model_id = "mlx-community/Qwen3.5-4B-MLX-4bit"
        dirty = "Answer. Thinking Process: 1. Analyze...\n\nFinal."
        result = engine._sanitize(dirty)
        assert "Thinking Process" not in result
        assert "Final." in result