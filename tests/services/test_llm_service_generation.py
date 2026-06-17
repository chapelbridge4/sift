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

import concurrent.futures

from app.services.llm_service import LLMService


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
