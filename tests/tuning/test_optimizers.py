"""
Unit tests for optimizer wrappers.
"""

from unittest.mock import MagicMock, patch

from app.tuning.optimizers import KVCacheOptimizer, PrefixCache, SpeculativeDecoder


class TestKVCacheOptimizer:
    """Test KV cache quantization optimizer."""

    def test_disabled_when_config_none(self):
        with patch("app.tuning.optimizers.get_settings") as mock_settings:
            mock_settings.return_value.KV_CACHE_QUANTIZATION = None
            opt = KVCacheOptimizer()
            assert not opt.enabled

    def test_enabled_when_q8(self):
        with patch("app.tuning.optimizers.get_settings") as mock_settings:
            mock_settings.return_value.KV_CACHE_QUANTIZATION = "q8"
            opt = KVCacheOptimizer()
            assert opt.enabled
            assert opt.quant_mode == "q8"

    def test_enabled_when_q4(self):
        with patch("app.tuning.optimizers.get_settings") as mock_settings:
            mock_settings.return_value.KV_CACHE_QUANTIZATION = "q4"
            opt = KVCacheOptimizer()
            assert opt.enabled
            assert opt.quant_mode == "q4"

    def test_wrap_generate_noop_when_disabled(self):
        with patch("app.tuning.optimizers.get_settings") as mock_settings:
            mock_settings.return_value.KV_CACHE_QUANTIZATION = None

            mock_fn = MagicMock(return_value="result")
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()

            opt = KVCacheOptimizer()
            result = opt.wrap_generate(mock_fn, mock_model, mock_tokenizer, prompt="test")

            mock_fn.assert_called_once()
            assert result == "result"

    def test_wrap_generate_logs_unsupported(self):
        with patch("app.tuning.optimizers.get_settings") as mock_settings:
            mock_settings.return_value.KV_CACHE_QUANTIZATION = "q8"
            mock_settings.return_value.BENCHMARK_OUTPUT_DIR = "/tmp/test_results"

            mock_fn = MagicMock()
            # Simulate mlx_vlm.generate without kv_cache_quantization param
            # Use spec=inspect to get a real signature object, not a MagicMock
            import inspect
            mock_fn = MagicMock(spec=inspect.signature(lambda x, **kw: None))

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()

            opt = KVCacheOptimizer()
            result = opt.wrap_generate(mock_fn, mock_model, mock_tokenizer, prompt="test")

            # Should still call the function (no crash)
            mock_fn.assert_called_once()


class TestSpeculativeDecoder:
    """Test speculative decoding optimizer."""

    def test_disabled_by_default(self):
        with patch("app.tuning.optimizers.get_settings") as mock_settings:
            mock_settings.return_value.SPECULATIVE_DECODING_ENABLED = False
            opt = SpeculativeDecoder()
            assert not opt.enabled

    def test_allowed_only_for_balanced_quality(self):
        with patch("app.tuning.optimizers.get_settings") as mock_settings:
            mock_settings.return_value.SPECULATIVE_DECODING_ENABLED = True
            mock_settings.return_value.SPECULATIVE_DRAFT_MODEL = "test-draft"
            mock_settings.return_value.SPECULATIVE_MIN_ACCEPTANCE_RATE = 0.5

            opt = SpeculativeDecoder()
            assert opt.is_allowed_for_profile("fast") is False
            assert opt.is_allowed_for_profile("balanced") is True
            assert opt.is_allowed_for_profile("quality") is True


class TestPrefixCache:
    """Test prefix caching optimizer."""

    def test_cache_key_deterministic(self):
        with patch("app.tuning.optimizers.get_settings") as mock_settings:
            mock_settings.return_value.PREFIX_CACHE_ENABLED = True
            mock_settings.return_value.PREFIX_CACHE_MAX_ENTRIES = 3

            opt = PrefixCache()
            key1 = opt.cache_key("system prompt", ["ctx1", "ctx2"])
            key2 = opt.cache_key("system prompt", ["ctx1", "ctx2"])
            key3 = opt.cache_key("different", ["ctx1"])

            assert key1 == key2
            assert key1 != key3

    def test_disabled_by_default(self):
        with patch("app.tuning.optimizers.get_settings") as mock_settings:
            mock_settings.return_value.PREFIX_CACHE_ENABLED = False

            opt = PrefixCache()
            assert not opt.enabled

    def test_max_entries_from_config(self):
        with patch("app.tuning.optimizers.get_settings") as mock_settings:
            mock_settings.return_value.PREFIX_CACHE_ENABLED = True
            mock_settings.return_value.PREFIX_CACHE_MAX_ENTRIES = 5

            opt = PrefixCache()
            assert opt.max_entries == 5


class TestRepetitionDetection:
    """Test repetition detection regex from benchmark module."""

    def test_detects_consecutive_word_repetition(self):
        from app.tuning.benchmark import detect_repetition

        assert detect_repetition("the the the the the the house")  # 5+ repeats
        assert detect_repetition("word word word word word word")  # 5+ repeats
        assert not detect_repetition("word word word")  # only 3 repeats — under threshold
        assert not detect_repetition("The quick brown fox")


class TestThinkingLeakDetection:
    """Test thinking leak detection from benchmark module."""

    def test_detects_thinking_process(self):
        from app.tuning.benchmark import detect_thinking_leak

        assert detect_thinking_leak("Thinking Process: 1. Analyze...")
        assert detect_thinking_leak("Thinking Process:")
        assert detect_thinking_leak("(思考中)...(思考完毕)")
        assert detect_thinking_leak("(思考中)")
        assert detect_thinking_leak("(思考完毕)")
        assert not detect_thinking_leak("This is a normal response without thinking")


class TestTokenCounting:
    """Test token counting heuristic."""

    def test_token_estimate(self):
        from app.tuning.benchmark import count_tokens

        # ~4 chars per token heuristic
        assert count_tokens("a" * 400, "test") == 100
        assert count_tokens("", "test") == 0


class TestDiagnosticFunctions:
    """Test diagnostics module."""

    def test_memory_snapshot_returns_dict(self):
        from app.tuning.diagnostics import memory_snapshot

        snapshot = memory_snapshot("test")
        assert isinstance(snapshot, dict)
        assert "rss_mb" in snapshot
        assert "vms_mb" in snapshot
        assert "available_gb" in snapshot
        assert snapshot["label"] == "test"

    def test_detect_swap_pressure_returns_bool(self):
        from app.tuning.diagnostics import detect_swap_pressure

        result = detect_swap_pressure()
        assert isinstance(result, bool)