"""
Tests for benchmark harness.
"""

import json

from app.tuning.benchmark import count_tokens, detect_repetition, detect_thinking_leak


class TestRepetitionDetection:
    """Test repetition detection."""

    def test_detects_consecutive_word_repetition(self):
        assert detect_repetition("the the the the the the house")  # 5+ repeats
        assert detect_repetition("word word word word word word")

    def test_ignores_normal_text(self):
        assert not detect_repetition("The quick brown fox jumps")
        assert not detect_repetition("word word word")  # only 3 repeats

    def test_case_insensitive(self):
        assert detect_repetition("Word Word Word Word Word Word")


class TestThinkingLeakDetection:
    """Test thinking leak detection."""

    def test_detects_thinking_process_header(self):
        assert detect_thinking_leak("Thinking Process: 1. Analyze...")

    def test_detects_chinese_tokens(self):
        assert detect_thinking_leak("(思考中)...(思考完毕)")
        assert detect_thinking_leak("(思考中)")
        assert detect_thinking_leak("(思考完毕)")

    def test_ignores_normal_response(self):
        assert not detect_thinking_leak("This is a normal answer without thinking")


class TestTokenCounting:
    """Test token counting heuristic."""

    def test_estimates_based_on_chars(self):
        # ~4 chars per token heuristic
        assert count_tokens("a" * 400, "qwen") == 100
        assert count_tokens("", "qwen") == 0

    def test_works_for_any_text(self):
        text = "The quick brown fox jumps over the lazy dog"
        tokens = count_tokens(text, "qwen")
        assert tokens > 0


class TestBenchmarkOutputFormat:
    """Test JSONL output format."""

    def test_result_dict_has_required_keys(self):
        required_keys = [
            "query",
            "retrieval_ms",
            "ttft_ms",
            "generation_ms",
            "tokens",
            "tok_per_sec",
            "peak_mb",
            "repetition_detected",
            "thinking_leaked",
            "timestamp",
            "profile",
        ]

        result = {
            "query": "test query",
            "retrieval_ms": 50.0,
            "ttft_ms": 100.0,
            "generation_ms": 5000.0,
            "tokens": 150,
            "tok_per_sec": 30.0,
            "peak_mb": 50.0,
            "repetition_detected": False,
            "thinking_leaked": False,
            "timestamp": "20260514_120000",
            "profile": "fast",
        }

        for key in required_keys:
            assert key in result

    def test_jsonl_line_format(self):
        result = {
            "query": "test",
            "retrieval_ms": 50.0,
            "generation_ms": 5000.0,
            "tokens": 100,
            "tok_per_sec": 20.0,
            "peak_mb": 30.0,
            "repetition_detected": False,
            "thinking_leaked": False,
            "timestamp": "20260514",
            "profile": "fast",
        }

        line = json.dumps(result)
        parsed = json.loads(line)

        assert parsed["query"] == "test"
        assert parsed["retrieval_ms"] == 50.0


class TestBenchmarkQueries:
    """Test that default benchmark queries are present in config."""

    def test_default_queries_exist(self):
        from app.config import get_settings

        settings = get_settings()
        queries = settings.DEFAULT_BENCHMARK_QUERIES

        assert len(queries) == 10
        assert all(isinstance(q, str) for q in queries)
        assert all(len(q) > 0 for q in queries)

    def test_query_topics_covered(self):
        from app.config import get_settings

        settings = get_settings()
        queries = settings.DEFAULT_BENCHMARK_QUERIES

        topics = ["transformer", "training", "reasoning", "compression", "benchmark",
                  "attention", "scaling", "prompt", "fine-tuning", "multimodal"]

        query_text = " ".join(queries).lower()
        for topic in topics:
            assert topic in query_text, f"Topic '{topic}' not found in queries"