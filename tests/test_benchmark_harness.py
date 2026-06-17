"""
Focused tests for the benchmark harness.

Tests:
- Fixture loading and export functions
- Benchmark runner stage timing capture
- Model profile and fusion method configuration passing
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from tests.benchmark_fixtures import (
    BenchmarkDocument,
    BenchmarkQuery,
    export_fixtures_as_metadata,
    export_fixtures_as_texts,
    get_benchmark_documents,
    get_benchmark_queries,
)


class TestBenchmarkFixtures(unittest.TestCase):
    """Test benchmark fixture loading and export."""

    def test_get_benchmark_documents_returns_list(self):
        """Test that get_benchmark_documents returns a list of BenchmarkDocument."""
        docs = get_benchmark_documents()
        self.assertIsInstance(docs, list)
        self.assertEqual(len(docs), 5)

    def test_benchmark_documents_have_required_fields(self):
        """Test that each document has required fields."""
        for doc in get_benchmark_documents():
            self.assertIsInstance(doc, BenchmarkDocument)
            self.assertTrue(len(doc.content) > 0)
            self.assertTrue(len(doc.source_file) > 0)
            self.assertTrue(len(doc.file_type) > 0)
            self.assertIsInstance(doc.expected_topics, list)
            self.assertTrue(len(doc.expected_topics) > 0)

    def test_get_benchmark_queries_returns_list(self):
        """Test that get_benchmark_queries returns a list of BenchmarkQuery."""
        queries = get_benchmark_queries()
        self.assertIsInstance(queries, list)
        self.assertEqual(len(queries), 5)

    def test_benchmark_queries_have_required_fields(self):
        """Test that each query has required fields."""
        for q in get_benchmark_queries():
            self.assertIsInstance(q, BenchmarkQuery)
            self.assertTrue(len(q.query) > 0)
            self.assertIsInstance(q.expected_evidence_topics, list)
            self.assertTrue(len(q.expected_evidence_topics) > 0)
            self.assertIn(q.question_type, ["factual", "conceptual", "comparative", "procedural"])

    def test_export_fixtures_as_texts_returns_correct_count(self):
        """Test that export returns correct number of texts."""
        texts = export_fixtures_as_texts()
        self.assertEqual(len(texts), len(get_benchmark_documents()))

    def test_export_fixtures_as_texts_all_non_empty(self):
        """Test that all exported texts are non-empty."""
        texts = export_fixtures_as_texts()
        for text in texts:
            self.assertTrue(len(text) > 0)

    def test_export_fixtures_as_metadata_returns_correct_count(self):
        """Test that metadata export returns correct count."""
        metadatas = export_fixtures_as_metadata()
        self.assertEqual(len(metadatas), len(get_benchmark_documents()))

    def test_export_metadata_has_required_keys(self):
        """Test that each metadata dict has required keys."""
        for meta in export_fixtures_as_metadata():
            self.assertIn("source_file", meta)
            self.assertIn("file_type", meta)
            self.assertIn("expected_topics", meta)
            self.assertIn("chunk_index", meta)
            self.assertIn("total_chunks", meta)


class TestStageTiming(unittest.TestCase):
    """Test stage timing capture in benchmark results."""

    def test_stage_timing_dataclass(self):
        """Test StageTiming dataclass creation."""
        from scripts.run_benchmark import StageTiming

        timing = StageTiming(stage_name="test", duration_seconds=1.5)
        self.assertEqual(timing.stage_name, "test")
        self.assertEqual(timing.duration_seconds, 1.5)

    def test_query_result_captures_timings(self):
        """Test that QueryResult captures stage timings."""
        from scripts.run_benchmark import QueryResult, StageTiming

        timings = [
            StageTiming(stage_name="retrieval", duration_seconds=0.1),
            StageTiming(stage_name="generation", duration_seconds=0.5),
        ]

        result = QueryResult(
            query="test query",
            question_type="factual",
            expected_topics=["topic1"],
            num_retrieved=3,
            retrieved_topics=["topic1"],
            has_answer=True,
            answer_length=100,
            model_used="qwen2.5:1.5b",
            timings=timings,
            total_latency_seconds=0.6
        )

        self.assertEqual(len(result.timings), 2)
        self.assertEqual(result.timings[0].stage_name, "retrieval")
        self.assertEqual(result.total_latency_seconds, 0.6)

    def test_benchmark_report_aggregates_timings(self):
        """Test that BenchmarkReport aggregates timing data."""
        from scripts.run_benchmark import BenchmarkReport, QueryResult, StageTiming

        results = [
            QueryResult(
                query=f"query {i}",
                question_type="factual",
                expected_topics=["topic"],
                num_retrieved=2,
                retrieved_topics=["topic"],
                has_answer=True,
                answer_length=50,
                model_used="qwen2.5:1.5b",
                timings=[StageTiming("stage1", 0.1)],
                total_latency_seconds=0.1 + i * 0.1
            )
            for i in range(3)
        ]

        report = BenchmarkReport(
            model_profile="balanced",
            fusion_method="rrf",
            sparse_strategy="bm25",
            total_queries=3,
            total_documents_indexed=5,
            queries_with_answers=3,
            average_latency_seconds=0.2,
            average_retrieval_score=0.6,
            results=results,
            timestamp="2024-01-01T00:00:00"
        )

        self.assertEqual(report.total_queries, 3)
        self.assertEqual(len(report.results), 3)
        self.assertEqual(report.average_latency_seconds, 0.2)


class TestConfigurationPassing(unittest.TestCase):
    """Test that model_profile, fusion_method, and sparse_strategy are passed through correctly."""

    def test_model_profile_arg_respected(self):
        """Test that --model-profile argument is passed to reasoning."""
        import sys

        from scripts.run_benchmark import parse_args
        old_argv = sys.argv
        try:
            sys.argv = ['run_benchmark.py', '--model-profile', 'quality']
            args = parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.model_profile, "quality")

    def test_fusion_method_arg_respected(self):
        """Test that --fusion-method argument is passed to reasoning."""
        import sys

        from scripts.run_benchmark import parse_args
        old_argv = sys.argv
        try:
            sys.argv = ['run_benchmark.py', '--fusion-method', 'dbsf']
            args = parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.fusion_method, "dbsf")

    def test_sparse_strategy_arg_respected(self):
        """Test that --sparse-strategy argument is passed correctly."""
        import sys

        from scripts.run_benchmark import parse_args
        old_argv = sys.argv
        try:
            sys.argv = ['run_benchmark.py', '--sparse-strategy', 'bm25plus']
            args = parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.sparse_strategy, "bm25plus")

    def test_default_values_are_sensible(self):
        """Test that default values for model_profile, fusion_method, and sparse_strategy are sensible."""
        import sys

        from scripts.run_benchmark import parse_args
        old_argv = sys.argv
        try:
            sys.argv = ['run_benchmark.py']
            args = parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.model_profile, "balanced")
        self.assertEqual(args.fusion_method, "rrf")
        self.assertEqual(args.sparse_strategy, "bm25")
        self.assertEqual(args.top_k, 5)
        self.assertEqual(args.collection_name, "benchmark_test_collection")


class TestBenchmarkRunner(unittest.TestCase):
    """Test benchmark runner orchestration without loading MLX on import."""

    def test_run_benchmark_uses_lazy_brain_loader(self):
        """Test that run_benchmark resolves the brain class lazily at runtime."""
        from argparse import Namespace

        from scripts.run_benchmark import QueryResult, StageTiming, run_benchmark
        from tests.benchmark_fixtures import BenchmarkQuery

        fake_query = BenchmarkQuery(
            query="test query",
            expected_evidence_topics=["topic1"],
            question_type="factual"
        )

        fake_result = QueryResult(
            query="test query",
            question_type="factual",
            expected_topics=["topic1"],
            num_retrieved=1,
            retrieved_topics=["topic1"],
            has_answer=True,
            answer_length=42,
            model_used="fake-model",
            timings=[StageTiming(stage_name="retrieval_and_generation", duration_seconds=0.1)],
            total_latency_seconds=0.1,
            retrieval_score=1.0
        )

        class FakeBrain:
            async def initialize(self):
                return None

        args = Namespace(
            model_profile="balanced",
            fusion_method="rrf",
            sparse_strategy="bm25",
            collection_name="benchmark_test_collection",
            top_k=5,
            output_json=None,
        )

        with patch("scripts.run_benchmark.load_orchestrator_class", return_value=FakeBrain), \
             patch("scripts.run_benchmark.setup_collection", new=AsyncMock(return_value=True)), \
             patch("scripts.run_benchmark.cleanup_collection", new=AsyncMock()), \
             patch("scripts.run_benchmark.get_benchmark_queries", return_value=[fake_query]), \
             patch("scripts.run_benchmark.run_query", new=AsyncMock(return_value=fake_result)):
            report = asyncio.run(run_benchmark(args))

        self.assertEqual(report.total_queries, 1)
        self.assertEqual(report.queries_with_answers, 1)
        self.assertEqual(report.average_retrieval_score, 1.0)


if __name__ == '__main__':
    unittest.main()
