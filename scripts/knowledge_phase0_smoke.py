#!/usr/bin/env python3
"""Phase 0 gate: smoke-test knowledge LLM on a small papers/ sample.

Measures JSON-schema validity, wall time, and basic claim output per paper.
Run ./scripts/hardware_guard.sh first (or pass --skip-hardware-guard for CI).

Usage:
    .venv/bin/python scripts/knowledge_phase0_smoke.py
    .venv/bin/python scripts/knowledge_phase0_smoke.py --papers 3 --profile papers
    .venv/bin/python scripts/knowledge_phase0_smoke.py --output reports/knowledge/phase0_smoke.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.knowledge.backend import build_knowledge_llm  # noqa: E402
from app.knowledge.config import load_profile  # noqa: E402
from app.knowledge.degraded import check_hardware_guard  # noqa: E402
from app.knowledge.tier0_cluster import extract_claim_spans  # noqa: E402
from app.knowledge.tier1_extract import extract_paper  # noqa: E402
from app.services.document_parser import DocumentParser  # noqa: E402

_DEFAULT_GUARD = ROOT / "scripts" / "hardware_guard.sh"


def _discover_papers(papers_dir: Path, limit: int) -> list[str]:
    paths = sorted(papers_dir.glob("*.pdf"))[:limit]
    if not paths:
        raise FileNotFoundError(f"no PDFs under {papers_dir}")
    return [str(p) for p in paths]


def _run_guard(skip: bool) -> None:
    if skip:
        return
    check_hardware_guard(_DEFAULT_GUARD)


async def _smoke_papers(
    file_paths: list[str],
    profile_name: str,
) -> dict:
    profile = load_profile(profile_name)
    parser = DocumentParser()
    llm = build_knowledge_llm(profile)

    parsed_docs = await parser.parse_for_knowledge(file_paths)
    results: list[dict] = []
    valid_json = 0
    t0 = time.perf_counter()

    for doc in parsed_docs:
        paper_t0 = time.perf_counter()
        spans = extract_claim_spans(doc, profile)
        paper_id = doc.paper_id
        err: str | None = None
        claim_count = 0
        try:
            summary = await extract_paper(doc, spans, llm, profile)
            valid_json += 1
            claim_count = len(summary.claims)
            degraded = summary.degraded
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            degraded = True

        results.append(
            {
                "paper_id": paper_id,
                "source_file": doc.source_file,
                "span_count": len(spans),
                "claim_count": claim_count,
                "degraded": degraded,
                "error": err,
                "wall_seconds": round(time.perf_counter() - paper_t0, 2),
            }
        )

    total_s = round(time.perf_counter() - t0, 2)
    n = len(results)
    validity_rate = valid_json / n if n else 0.0

    return {
        "profile": profile_name,
        "model_id": profile.llm.model_id,
        "papers_tested": n,
        "valid_json_count": valid_json,
        "validity_rate": round(validity_rate, 4),
        "total_wall_seconds": total_s,
        "budget_seconds": 900,
        "within_budget": total_s < 900,
        "target_validity_rate": 0.95,
        "passed": validity_rate >= 0.95 and valid_json == n,
        "per_paper": results,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 0 knowledge LLM smoke test")
    p.add_argument(
        "--papers-dir",
        default="papers",
        help="Directory with PDF corpus (default: papers/)",
    )
    p.add_argument("--papers", type=int, default=3, help="Number of PDFs to test (default: 3)")
    p.add_argument("--profile", default="papers", help="Knowledge profile name")
    p.add_argument(
        "--skip-hardware-guard",
        action="store_true",
        help="Skip RAM pre-flight (tests only)",
    )
    p.add_argument(
        "--output",
        default="",
        help="Optional JSON report path (e.g. reports/knowledge/phase0_smoke.json)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    papers_dir = Path(args.papers_dir)
    if not papers_dir.is_absolute():
        papers_dir = ROOT / papers_dir

    try:
        _run_guard(args.skip_hardware_guard)
        file_paths = _discover_papers(papers_dir, args.papers)
    except Exception as exc:
        print(f"PHASE0 ABORT: {exc}")
        return 1

    print(f"Phase 0 smoke: {len(file_paths)} papers, profile={args.profile}")
    report = asyncio.run(_smoke_papers(file_paths, args.profile))

    print(json.dumps(report, indent=2))
    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}")

    if report["passed"]:
        print("PHASE0 OK: validity rate >= 95%, all papers succeeded")
        return 0

    print(
        f"PHASE0 FAIL: validity={report['validity_rate']:.1%} "
        f"({report['valid_json_count']}/{report['papers_tested']})"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())