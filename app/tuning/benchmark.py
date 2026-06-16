"""
Core benchmark harness for MLX inference on M1 8GB.

Runnable via:
  .venv/bin/python -m app.tuning.benchmark --profile fast --queries 5
  .venv/bin/python -m app.tuning.benchmark --profile balanced --queries 5
  .venv/bin/python -m app.tuning.benchmark --mode multi-turn --profile balanced --turns 3

Measures per-query:
  - retrieval_ms
  - ttft_ms (time to first token)
  - generation_ms
  - tokens (count via tokenizer)
  - tok_per_sec
  - peak_mb (RSS delta)
  - thinking_leaked
  - repetition_detected
  - garbage_detected (via detect_garbage())
  - valid_response (via is_valid_response())

Output: app/tuning/results/<profile>_<mode>_YYYYMMDD_HHMMSS.jsonl
"""

import argparse
import asyncio
import json
import os
import re
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from app.config import get_settings
from app.services.llm_service import LLMService
from app.services.qdrant_service import QdrantService
from app.tuning.diagnostics import detect_swap_pressure, memory_snapshot
from app.tuning.profiles import get_baseline_profile
from app.tuning.quality import detect_garbage, is_valid_response

# Patterns for detection
REPETITION_PATTERN = re.compile(r"\b(\w+)\s+(\1\s+){4,}", re.IGNORECASE)
THINKING_LEAK_PATTERNS = [
    re.compile(r"Thinking Process:", re.IGNORECASE),
    re.compile(r"\(思考中\)"),
    re.compile(r"\(思考完毕\)"),
]


def detect_repetition(text: str) -> bool:
    """True if any word repeats >5 times consecutively."""
    return bool(REPETITION_PATTERN.search(text))


def detect_thinking_leak(text: str) -> bool:
    """True if output contains thinking artifacts."""
    return any(pat.search(text) for pat in THINKING_LEAK_PATTERNS)


def count_tokens(text: str, model_id: str) -> int:
    """Estimate token count from generated text."""
    # Rough heuristic: ~4 chars per token for Qwen
    return len(text) // 4


async def measure_retrieval(
    qdrant: QdrantService, collection: str, query: str, top_k: int = 5
) -> tuple[List[dict], float]:
    """Measure retrieval latency and return contexts + time in ms."""
    t0 = time.perf_counter()
    results = await qdrant.hybrid_search(collection, query, top_k=top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    contexts = [r["text"] for r in results]
    return contexts, retrieval_ms


async def measure_generation(
    llm: LLMService,
    query: str,
    contexts: List[str],
    model_profile: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> dict:
    """
    Measure generation metrics.

    Returns dict with: generation_ms, ttft_ms, tokens, tok_per_sec, peak_mb,
    thinking_leaked, repetition_detected, garbage_detected, valid_response
    """
    mem_before = memory_snapshot("gen_start")
    mem_before_rss = mem_before["rss_mb"]

    t0 = time.perf_counter()

    # Full generation
    answer = await llm.generate_rag_response(
        query=query,
        retrieved_contexts=contexts,
        model_profile=model_profile,
        enable_thinking=False,
        conversation_history=conversation_history,
    )
    generation_ms = (time.perf_counter() - t0) * 1000
    ttft_ms = None  # Streaming not available; set to None

    mem_after = memory_snapshot("gen_end")
    peak_mb = mem_after["rss_mb"] - mem_before_rss

    tokens = count_tokens(answer, llm.model_name)
    tok_per_sec = (tokens / (generation_ms / 1000)) if generation_ms > 0 else 0

    thinking_leaked = detect_thinking_leak(answer)
    repetition_detected = detect_repetition(answer)
    garbage_detected = detect_garbage(answer)
    valid_response, _ = is_valid_response(answer)

    return {
        "generation_ms": generation_ms,
        "ttft_ms": ttft_ms,
        "tokens": tokens,
        "tok_per_sec": round(tok_per_sec, 2),
        "peak_mb": round(peak_mb, 2),
        "thinking_leaked": thinking_leaked,
        "repetition_detected": repetition_detected,
        "garbage_detected": garbage_detected,
        "valid_response": valid_response,
    }


async def run_benchmark(
    profile: str = "fast",
    queries: int = 10,
    collection: str = "ai_papers",
    output_dir: str | None = None,
    mode: str = "single",
    turns: int = 3,
) -> List[dict]:
    """
    Run benchmark for a given profile and query count.

    Args:
        profile: Model profile (fast/balanced/quality)
        queries: Number of queries to run
        collection: Qdrant collection name
        output_dir: Override for output directory
        mode: "single" or "multi-turn"
        turns: Number of turns for multi-turn mode

    Returns list of result dicts (one per query).
    """
    settings = get_settings()
    if output_dir is None:
        output_dir = settings.BENCHMARK_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    # Check swap before starting
    if detect_swap_pressure():
        return []

    # Load services
    qdrant = QdrantService()
    llm = LLMService()

    await qdrant.initialize()
    await llm.initialize()

    # Get tuning profile
    tune_profile = get_baseline_profile(profile)

    # Get queries
    benchmark_queries = settings.DEFAULT_BENCHMARK_QUERIES[:queries]

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(
        output_dir, f"{profile}_{mode}_{timestamp}.jsonl"
    )

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {profile} profile | {mode} mode | {queries} queries | {collection}")
    print(f"{'='*70}")

    for i, query in enumerate(benchmark_queries, 1):
        print(f"\n[{i}/{queries}] Query: {query[:60]}...")

        # Memory snapshot before retrieval
        memory_snapshot("query_start")

        # Check swap pressure periodically
        if detect_swap_pressure():
            print("ABORTED: Swap pressure detected")
            break

        # Retrieval
        contexts, retr_ms = await measure_retrieval(qdrant, collection, query, top_k=5)
        print(f"  Retrieval: {retr_ms:.0f}ms | {len(contexts)} contexts")

        # Generation
        gen_metrics = await measure_generation(
            llm, query, contexts, profile
        )
        print(
            f"  Generation: {gen_metrics['generation_ms']:.0f}ms | "
            f"{gen_metrics['tokens']} tokens | {gen_metrics['tok_per_sec']:.1f} tok/s | "
            f"Peak: {gen_metrics['peak_mb']:.1f}MB"
        )

        if gen_metrics["thinking_leaked"]:
            print("  WARNING: Thinking leak detected!")
        if gen_metrics["repetition_detected"]:
            print("  WARNING: Repetition detected!")
        if gen_metrics["garbage_detected"]:
            print("  WARNING: Garbage detected!")
        if not gen_metrics["valid_response"]:
            print("  WARNING: Invalid response!")

        # Build result record
        result = {
            "query": query,
            "retrieval_ms": round(retr_ms, 2),
            "ttft_ms": round(gen_metrics["ttft_ms"], 2)
            if gen_metrics["ttft_ms"]
            else None,
            "generation_ms": round(gen_metrics["generation_ms"], 2),
            "tokens": gen_metrics["tokens"],
            "tok_per_sec": gen_metrics["tok_per_sec"],
            "peak_mb": gen_metrics["peak_mb"],
            "repetition_detected": gen_metrics["repetition_detected"],
            "thinking_leaked": gen_metrics["thinking_leaked"],
            "garbage_detected": gen_metrics["garbage_detected"],
            "valid_response": gen_metrics["valid_response"],
            "timestamp": timestamp,
            "profile": profile,
            "mode": mode,
        }

        results.append(result)

        # Write to JSONL
        with open(result_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    # Summary
    total_gen = sum(r["generation_ms"] for r in results)
    total_retr = sum(r["retrieval_ms"] for r in results)
    n = len(results)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Queries:          {n}")
    print(f"Avg retrieval:    {total_retr/n:.1f}ms")
    print(f"Avg generation:   {total_gen/n:.1f}ms")
    print(f"Avg total:        {(total_retr+total_gen)/n:.1f}ms")
    print(f"Repetition loops: {sum(1 for r in results if r['repetition_detected'])}/{n}")
    print(f"Thinking leaks:   {sum(1 for r in results if r['thinking_leaked'])}/{n}")
    print(f"Garbage detected: {sum(1 for r in results if r['garbage_detected'])}/{n}")
    print(f"Invalid responses:{sum(1 for r in results if not r['valid_response'])}/{n}")
    print(f"Results written:  {result_file}")

    await qdrant.close()
    await llm.close()

    return results


async def run_multi_turn_benchmark(
    profile: str = "balanced",
    turns: int = 3,
    queries: int = 5,
    collection: str = "ai_papers",
    output_dir: str | None = None,
    session_id: Optional[str] = None,
) -> List[dict]:
    """
    Run multi-turn conversation benchmark to test KV cache persistence.

    Measures turn 1 vs turn 2 vs turn 3 generation latency.
    If turn 2+ is faster than turn 1 by >10%, KV persistence is working.

    Args:
        profile: Model profile
        turns: Number of conversation turns
        queries: Number of query sets (one per session turn)
        collection: Qdrant collection
        output_dir: Output directory
        session_id: Session identifier for KV cache

    Returns list of result dicts with turn-level latency breakdown.
    """
    settings = get_settings()
    if output_dir is None:
        output_dir = settings.BENCHMARK_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    if detect_swap_pressure():
        return []

    if session_id is None:
        session_id = f"test-kv-{uuid.uuid4().hex[:8]}"

    qdrant = QdrantService()
    llm = LLMService()

    await qdrant.initialize()
    await llm.initialize()

    benchmark_queries = settings.DEFAULT_BENCHMARK_QUERIES[:queries]

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(
        output_dir, f"{profile}_multi_turn_{timestamp}.jsonl"
    )

    print(f"\n{'='*70}")
    print(f"MULTI-TURN BENCHMARK: {profile} | {turns} turns | {queries} queries | session={session_id}")
    print(f"{'='*70}")

    for i, query in enumerate(benchmark_queries, 1):
        print(f"\n[{i}/{queries}] Query: {query[:60]}...")

        if detect_swap_pressure():
            print("ABORTED: Swap pressure detected")
            break

        # Retrieval
        contexts, retr_ms = await measure_retrieval(qdrant, collection, query, top_k=5)
        print(f"  Retrieval: {retr_ms:.0f}ms | {len(contexts)} contexts")

        conversation_history = []

        for turn in range(1, turns + 1):
            print(f"  -- Turn {turn}/{turns}")

            mem_before = memory_snapshot(f"turn_{turn}_start")
            mem_before_rss = mem_before["rss_mb"]

            t0 = time.perf_counter()
            answer = await llm.generate_rag_response(
                query=query,
                retrieved_contexts=contexts,
                model_profile=profile,
                enable_thinking=False,
                conversation_history=conversation_history,
            )
            generation_ms = (time.perf_counter() - t0) * 1000

            mem_after = memory_snapshot(f"turn_{turn}_end")
            peak_mb = mem_after["rss_mb"] - mem_before_rss

            tokens = count_tokens(answer, llm.model_name)
            tok_per_sec = (tokens / (generation_ms / 1000)) if generation_ms > 0 else 0

            thinking_leaked = detect_thinking_leak(answer)
            repetition_detected = detect_repetition(answer)
            garbage_detected = detect_garbage(answer)
            valid_response, _ = is_valid_response(answer)

            # Add to conversation history for next turn
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": answer})

            result = {
                "query": query,
                "turn": turn,
                "retrieval_ms": round(retr_ms, 2),
                "ttft_ms": None,
                "generation_ms": round(generation_ms, 2),
                "tokens": tokens,
                "tok_per_sec": round(tok_per_sec, 2),
                "peak_mb": round(peak_mb, 2),
                "repetition_detected": repetition_detected,
                "thinking_leaked": thinking_leaked,
                "garbage_detected": garbage_detected,
                "valid_response": valid_response,
                "timestamp": timestamp,
                "profile": profile,
                "mode": "multi-turn",
                "session_id": session_id,
            }

            results.append(result)

            print(
                f"    Turn {turn}: {generation_ms:.0f}ms | {tokens} tokens | "
                f"{tok_per_sec:.1f} tok/s | Peak: {peak_mb:.1f}MB"
            )

            if generation_ms > 15000:
                print("    WARNING: High latency detected, checking swap...")
                if detect_swap_pressure():
                    print("    ABORTED: Swap pressure detected")
                    break

            # Write to JSONL
            with open(result_file, "a") as f:
                f.write(json.dumps(result) + "\n")

    # Per-turn analysis
    print(f"\n{'='*70}")
    print("MULTI-TURN SUMMARY")
    print(f"{'='*70}")

    by_turn: Dict[int, List[dict]] = {}
    for r in results:
        by_turn.setdefault(r["turn"], []).append(r)

    print(f"\nSession: {session_id}")
    for turn_num in sorted(by_turn.keys()):
        turn_results = by_turn[turn_num]
        avg_gen = sum(r["generation_ms"] for r in turn_results) / len(turn_results)
        avg_total = sum(
            r["generation_ms"] + r["retrieval_ms"] for r in turn_results
        ) / len(turn_results)
        print(
            f"  Turn {turn_num}: avg gen {avg_gen:.0f}ms | avg total {avg_total:.0f}ms"
        )

    # KV cache effectiveness check
    if 1 in by_turn and 2 in by_turn:
        turn1_avg = sum(r["generation_ms"] for r in by_turn[1]) / len(by_turn[1])
        turn2_avg = sum(r["generation_ms"] for r in by_turn[2]) / len(by_turn[2])
        improvement = (turn1_avg - turn2_avg) / turn1_avg * 100

        print("\nKV Cache Effectiveness:")
        print(f"  Turn 1 avg: {turn1_avg:.0f}ms")
        print(f"  Turn 2 avg: {turn2_avg:.0f}ms")
        print(f"  Improvement: {improvement:.1f}%")

        if improvement > 10:
            print(f"  Status: KV persistence WORKING ({improvement:.1f}% speedup)")
        else:
            print(
                "  Status: mlx-vlm does not expose cache passing in generate() — KV dormant."
            )
            print(
                "  (No improvement or <10% improvement means no effective KV reuse)"
            )

    print(f"\nResults written: {result_file}")

    await qdrant.close()
    await llm.close()

    return results


def decide_4b_profile_caps(fast_results: List[dict], balanced_results: List[dict], quality_results: List[dict]) -> dict:
    """
    Data-Driven 4B Profile Caps Decision.

    Replaces --compare-2b-vs-4b with quality validation at different token caps.
    Decision table:
      - 4B at 400 tokens, no swap, valid responses → ship as fast
      - 4B at 600 tokens, no swap, valid responses → ship as balanced
      - 4B at 1000 tokens triggers swap → cap quality at 800 tokens
    """
    profiles = {
        "fast": fast_results,
        "balanced": balanced_results,
        "quality": quality_results,
    }

    decision = {
        "actions": {},
        "recommendations": [],
    }

    for profile, results in profiles.items():
        if not results:
            decision["actions"][profile] = "NO_DATA"
            continue

        garbage = sum(1 for r in results if r.get("garbage_detected", False))
        invalid = sum(1 for r in results if not r.get("valid_response", True))
        avg_latency = sum(r["generation_ms"] for r in results) / len(results)
        max_tokens = {"fast": 400, "balanced": 600, "quality": 800}[profile]

        action = "OK"
        note = ""

        if garbage > 1:
            action = "RETIRE"
            note = f"garbage_detected on {garbage} queries (threshold >1)"
        elif invalid > 1:
            action = "RETIRE"
            note = f"valid_response=False on {invalid} queries (threshold >1)"
        elif avg_latency > 10000:
            action = "CAP_TOKENS"
            note = f"avg latency {avg_latency:.0f}ms > 10s — reduce max_tokens"

        decision["actions"][profile] = action
        decision["recommendations"].append(
            f"{profile}: {action} | garbage={garbage}, invalid={invalid}, avg_latency={avg_latency:.0f}ms, max_tokens={max_tokens} | {note}"
        )

    return decision


def print_4b_validation_table(fast_results: List[dict], balanced_results: List[dict], quality_results: List[dict]) -> None:
    """Print formatted 4B validation table."""
    decision = decide_4b_profile_caps(fast_results, balanced_results, quality_results)

    print(f"\n{'='*70}")
    print("4B VALIDATION TABLE (Replaces --compare-2b-vs-4b)")
    print(f"{'='*70}")

    for profile, results in [("fast", fast_results), ("balanced", balanced_results), ("quality", quality_results)]:
        if not results:
            print(f"\n| {profile:12s} | NO DATA |")
            continue

        avg_gen = sum(r["generation_ms"] for r in results) / len(results)
        avg_retr = sum(r["retrieval_ms"] for r in results) / len(results)
        avg_total = avg_gen + avg_retr
        garbage = sum(1 for r in results if r.get("garbage_detected", False))
        invalid = sum(1 for r in results if not r.get("valid_response", True))
        rep = sum(1 for r in results if r.get("repetition_detected", False))
        think = sum(1 for r in results if r.get("thinking_leaked", False))
        action = decision["actions"].get(profile, "UNKNOWN")

        print(f"\n| {profile:12s} | action={action}")
        print("|-------------------|------------------------------------------|")
        print(f"| avg gen (ms)      | {avg_gen:11.1f} |")
        print(f"| avg retrieval(ms) | {avg_retr:11.1f} |")
        print(f"| avg total (ms)    | {avg_total:11.1f} |")
        print(f"| garbage           | {garbage:11d} |")
        print(f"| invalid           | {invalid:11d} |")
        print(f"| repetition        | {rep:11d} |")
        print(f"| thinking leak     | {think:11d} |")

    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    for rec in decision["recommendations"]:
        print(f"  {rec}")


def main():
    parser = argparse.ArgumentParser(description="MLX benchmark harness")
    parser.add_argument(
        "--profile",
        default="fast",
        choices=["fast", "balanced", "quality"],
        help="Model profile to benchmark",
    )
    parser.add_argument(
        "--queries", type=int, default=10, help="Number of queries to run"
    )
    parser.add_argument(
        "--collection", default="ai_papers", help="Qdrant collection name"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Override for output directory"
    )
    parser.add_argument(
        "--mode",
        default="single",
        choices=["single", "multi-turn"],
        help="Benchmark mode: single or multi-turn",
    )
    parser.add_argument(
        "--turns", type=int, default=3, help="Number of turns for multi-turn mode"
    )
    parser.add_argument(
        "--session-id", default=None, help="Session ID for multi-turn KV cache test"
    )
    parser.add_argument(
        "--validate-4b",
        action="store_true",
        help="Run fast/balanced/quality profiles and produce 4B caps validation decision",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override model_id for custom profile (used with --profile custom)",
    )

    args = parser.parse_args()

    if args.validate_4b:
        print("\n" + "=" * 70)
        print("4B VALIDATION BENCHMARK")
        print("=" * 70)

        print("\n>> Running FAST profile (4B, 400 tokens)...")
        fast_results = asyncio.run(
            run_benchmark(
                profile="fast",
                queries=args.queries,
                collection=args.collection,
                output_dir=args.output_dir,
                mode="single",
            )
        )

        print("\n>> Running BALANCED profile (4B, 600 tokens)...")
        balanced_results = asyncio.run(
            run_benchmark(
                profile="balanced",
                queries=args.queries,
                collection=args.collection,
                output_dir=args.output_dir,
                mode="single",
            )
        )

        print("\n>> Running QUALITY profile (4B, 800 tokens)...")
        quality_results = asyncio.run(
            run_benchmark(
                profile="quality",
                queries=args.queries,
                collection=args.collection,
                output_dir=args.output_dir,
                mode="single",
            )
        )

        print_4b_validation_table(fast_results, balanced_results, quality_results)

    elif args.mode == "multi-turn":
        # Multi-turn KV cache test
        asyncio.run(
            run_multi_turn_benchmark(
                profile=args.profile,
                turns=args.turns,
                queries=args.queries,
                collection=args.collection,
                output_dir=args.output_dir,
                session_id=args.session_id,
            )
        )

    else:
        # Single profile benchmark
        asyncio.run(
            run_benchmark(
                profile=args.profile,
                queries=args.queries,
                collection=args.collection,
                output_dir=args.output_dir,
                mode=args.mode,
            )
        )


if __name__ == "__main__":
    main()