"""
Generation-path regression tests for LLMService.

These are CHEAP (no model load) and guard two fixed bugs:

Bug A — the RAG system prompt must NOT be prefixed with "/no_think". On this
MLX build that prefix lands at the start of the raw user turn and produces
malformed output (leaked empty <think></think> block / empty string). Thinking
is disabled via the model default + the output sanitizer instead.

Bug C — every MLX-Metal GPU op is thread-affine. LLMService must own ONE
dedicated single-worker executor so load/warmup/generate all run on the same
thread, avoiding "There is no Stream(gpu, 1) in current thread."
"""

import asyncio
import concurrent.futures
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.llm_service import LLMService


def test_validity_gate_strict_when_not_thinking():
    """enable_thinking=False keeps the strict gate: a too-short answer is
    rejected exactly as is_valid_response decides."""
    valid, reason = LLMService._passes_validity("ok", enable_thinking=False)
    assert valid is False
    assert reason == "too_short"


def test_validity_gate_accepts_short_thinking_answer():
    """enable_thinking=True: a short-but-NON-EMPTY answer is accepted even if
    the strict gate would call it too_short (thinking ate the token budget)."""
    valid, reason = LLMService._passes_validity("Yes.", enable_thinking=True)
    assert valid is True


def test_validity_gate_rejects_empty_thinking_answer():
    """enable_thinking=True: a truly empty answer is still rejected."""
    valid, reason = LLMService._passes_validity("   ", enable_thinking=True)
    assert valid is False
    assert reason == "too_short"


def test_rag_system_prompt_has_no_no_think_prefix():
    """Bug A: the system prompt built for a Qwen model must not start with
    the '/no_think' hack. Uses the pure builder seam so no model is loaded."""
    svc = LLMService()
    system_prompt = svc._build_rag_system_prompt("mlx-community/Qwen3.5-4B-MLX-4bit")
    assert not system_prompt.lstrip().startswith("/no_think"), (
        "RAG system prompt must not inject the '/no_think' prefix; "
        f"got: {system_prompt[:40]!r}"
    )
    # The real instruction must survive.
    assert "provided context" in system_prompt


def test_llm_service_owns_single_worker_mlx_executor():
    """Bug C: LLMService pins all MLX ops to one thread via a dedicated
    single-worker executor."""
    svc = LLMService()
    ex = svc._get_mlx_executor()
    assert isinstance(ex, concurrent.futures.ThreadPoolExecutor)
    assert ex._max_workers == 1
    # Same executor instance is reused (one thread for the lifetime).
    assert svc._get_mlx_executor() is ex


class _Chunk:
    def __init__(self, text):
        self.text = text


def _collect(agen):
    async def run():
        return [chunk async for chunk in agen]
    return asyncio.run(run())


def _service_for_stream():
    svc = LLMService()
    svc.initialize = AsyncMock()
    svc._current_model = (MagicMock(), MagicMock())
    svc.current_config = {"max_tokens": 16}
    return svc


def test_stream_generate_runs_on_mlx_thread_and_yields():
    """Bug-C class: streaming must run on the dedicated MLX thread (not the
    event-loop thread) and yield chunks in order."""
    svc = _service_for_stream()
    seen_threads = []

    def fake_stream(model, tokenizer, **kwargs):
        seen_threads.append(threading.current_thread().name)
        yield _Chunk("Hello ")
        yield _Chunk("world")

    try:
        with patch("mlx_vlm.stream_generate", fake_stream):
            out = _collect(svc.stream_generate("hi"))
        assert out == ["Hello ", "world"]
        assert seen_threads and seen_threads[0].startswith("mlx"), (
            f"stream must run on the MLX executor thread, ran on {seen_threads}"
        )
    finally:
        svc.close()


def test_stream_generate_propagates_errors():
    """An error raised inside the MLX stream surfaces to the async caller."""
    svc = _service_for_stream()

    def boom(model, tokenizer, **kwargs):
        yield _Chunk("partial")
        raise RuntimeError("stream blew up")

    try:
        with patch("mlx_vlm.stream_generate", boom):
            with pytest.raises(RuntimeError, match="stream blew up"):
                _collect(svc.stream_generate("hi"))
    finally:
        svc.close()
