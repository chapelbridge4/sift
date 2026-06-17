"""CLI for offline corpus-intelligence builds (`python -m app.knowledge build`)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.kbforge.discover.scanner import scan_directory
from app.knowledge import __version__
from app.knowledge.backend import KnowledgeLLM, get_knowledge_backend
from app.knowledge.config import load_profile
from app.knowledge.index import index_artifacts
from app.knowledge.pipeline import KnowledgePipeline
from app.services.document_parser import DocumentParser
from app.services.embeddings import EmbeddingService
from app.services.qdrant_service import QdrantService

_KNOWLEDGE_EXTENSIONS = ["pdf", "docx", "txt", "md", "html"]


def _setup_logging(verbose: bool, quiet: bool) -> None:
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="[knowledge] %(message)s")


def scan_input_files(input_dir: Path) -> list[str]:
    """Discover ingestible files under --input (reuses kbforge scanner)."""
    sources = scan_directory(input_dir.resolve(), _KNOWLEDGE_EXTENSIONS)
    return [doc.path for doc in sources]


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).expanduser()
    settings = get_settings()
    if settings.KNOWLEDGE_OUTPUT_DIR:
        return Path(settings.KNOWLEDGE_OUTPUT_DIR).expanduser()
    return Path(settings.ALLOWED_CORPUS_DIR) / ".knowledge"


def cmd_build(args: argparse.Namespace) -> int:
    input_dir = Path(args.input).expanduser()
    if not input_dir.is_dir():
        logging.error("input directory does not exist: %s", input_dir)
        return 1

    file_paths = scan_input_files(input_dir)
    if not file_paths:
        logging.error("no ingestible files found under %s", input_dir)
        return 1

    profile = load_profile(args.profile)
    output_dir = _resolve_output_dir(args)
    artifact_dir = output_dir / args.collection

    pipeline = KnowledgePipeline(
        parser=DocumentParser(),
        embedder=EmbeddingService(),
        llm=KnowledgeLLM(get_knowledge_backend(profile)),
        profile=profile,
        output_dir=output_dir,
        skip_hardware_guard=args.skip_hardware_guard,
    )

    stats = asyncio.run(pipeline.run(file_paths, args.collection))

    indexed = 0
    if not args.skip_index:
        qdrant = QdrantService()
        asyncio.run(qdrant.initialize())
        indexed = asyncio.run(
            index_artifacts(
                collection_name=args.collection,
                artifact_dir=artifact_dir,
                profile=profile,
                qdrant_service=qdrant,
            )
        )
        stats = stats.model_copy(update={"chunks": indexed})

    print(f"profile={profile.name} collection={args.collection}")
    print(f"discovered {len(file_paths)} files")
    print(f"papers={stats.papers} topics={stats.topics} links={stats.links}")
    print(f"artifact chunks indexed={indexed}")
    print(f"artifacts → {artifact_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knowledge",
        description="Corpus intelligence builder (topic sheets + paper summaries)",
    )
    parser.add_argument("--version", action="version", version=f"knowledge {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build knowledge artifacts and index into Qdrant")
    build.add_argument("--input", required=True, help="Source directory")
    build.add_argument("--collection", required=True, help="Qdrant collection name")
    build.add_argument(
        "--profile",
        default="papers",
        help="Knowledge profile name (knowledge_<name>.toml)",
    )
    build.add_argument(
        "--output",
        default=None,
        help="Artifact output root (default: {ALLOWED_CORPUS_DIR}/.knowledge)",
    )
    build.add_argument(
        "--skip-index",
        action="store_true",
        help="Generate artifacts only; do not upsert into Qdrant",
    )
    build.add_argument(
        "--skip-hardware-guard",
        action="store_true",
        help="Skip RAM pre-flight check (tests / CI only)",
    )
    build.set_defaults(
        func=cmd_build,
        skip_index=False,
        skip_hardware_guard=False,
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose, args.quiet)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())