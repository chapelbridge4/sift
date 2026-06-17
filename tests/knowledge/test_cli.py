"""CLI argument parsing and build command wiring (pipeline.run mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.knowledge.cli import build_parser, cmd_build, main
from app.knowledge.models import KnowledgeStats


def test_build_parser_requires_input_and_collection():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["build"])

    args = parser.parse_args(
        ["build", "--input", "./papers", "--collection", "ai_papers_knowledge"]
    )
    assert args.command == "build"
    assert args.input == "./papers"
    assert args.collection == "ai_papers_knowledge"
    assert args.profile == "papers"
    assert args.output is None


def test_build_parser_accepts_profile_and_output():
    parser = build_parser()
    args = parser.parse_args(
        [
            "build",
            "--input",
            "/tmp/corpus",
            "--collection",
            "my_coll",
            "--profile",
            "papers",
            "--output",
            "/tmp/out/.knowledge",
        ]
    )
    assert args.profile == "papers"
    assert args.output == "/tmp/out/.knowledge"


@patch("app.knowledge.cli.QdrantService")
@patch("app.knowledge.cli.KnowledgePipeline")
@patch("app.knowledge.cli.scan_input_files")
def test_cmd_build_runs_pipeline_and_indexes(
    mock_scan,
    mock_pipeline_cls,
    mock_qdrant_cls,
    tmp_path,
):
    input_dir = tmp_path / "corpus"
    input_dir.mkdir()
    (input_dir / "a.pdf").write_bytes(b"%PDF-1.4")

    mock_scan.return_value = [str(input_dir / "a.pdf")]

    mock_pipeline = MagicMock()
    mock_pipeline.run = AsyncMock(
        return_value=KnowledgeStats(topics=2, papers=3, chunks=0, links=4)
    )
    mock_pipeline_cls.return_value = mock_pipeline

    mock_qdrant = MagicMock()
    mock_qdrant.initialize = AsyncMock()
    mock_qdrant_cls.return_value = mock_qdrant

    output_dir = tmp_path / "artifacts"

    with patch("app.knowledge.cli.index_artifacts", new_callable=AsyncMock) as mock_index:
        mock_index.return_value = 12

        args = MagicMock()
        args.input = str(input_dir)
        args.collection = "test_coll"
        args.profile = "papers"
        args.output = str(output_dir)
        args.skip_index = False
        args.skip_hardware_guard = True

        code = cmd_build(args)

    assert code == 0
    from app.knowledge.config import load_profile

    mock_scan.assert_called_once_with(Path(str(input_dir)), load_profile("papers"))
    mock_pipeline_cls.assert_called_once()
    mock_pipeline.run.assert_awaited_once_with(
        [str(input_dir / "a.pdf")],
        "test_coll",
    )
    mock_index.assert_awaited_once()
    call_kwargs = mock_index.await_args.kwargs
    assert call_kwargs["collection_name"] == "test_coll"
    assert call_kwargs["artifact_dir"] == output_dir / "test_coll"


@patch("app.knowledge.cli.cmd_build", return_value=0)
def test_main_dispatches_build_subcommand(mock_cmd_build):
    code = main(
        [
            "build",
            "--input",
            "./papers",
            "--collection",
            "ai_papers_knowledge",
            "--profile",
            "papers",
        ]
    )
    assert code == 0
    mock_cmd_build.assert_called_once()


def test_cmd_build_fails_fast_on_empty_input(tmp_path):
    args = MagicMock()
    args.input = str(tmp_path / "empty")
    args.collection = "coll"
    args.profile = "papers"
    args.output = None
    args.skip_index = False
    args.skip_hardware_guard = True

    with patch("app.knowledge.cli.scan_input_files", return_value=[]):
        code = cmd_build(args)

    assert code == 1