#!/usr/bin/env python3
"""Live HTTP smoke for make_knowledge via FastAPI /query and /collections.

Spins up uvicorn (unless --base-url is set), hits the real REST API with
use_llm=false (retrieval-only — no MLX/GGUF generation load), and validates
knowledge metadata + drill_down behavior.

Usage:
    .venv/bin/python scripts/knowledge_api_smoke.py

    # Against an already-running server:
    .venv/bin/python scripts/knowledge_api_smoke.py --base-url http://127.0.0.1:8000

    .venv/bin/python scripts/knowledge_api_smoke.py --collection ai_papers_knowledge \\
        --output reports/knowledge/api_smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.kbforge.eval.keyword_recall import compute_keyword_recall  # noqa: E402
from app.kbforge.eval.probe_evaluator import load_probes  # noqa: E402

_DEFAULT_PORT = 18765
_STARTUP_TIMEOUT_S = 90


def _wait_for_health(base_url: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_err = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}/health", timeout=5) as resp:
                return json.loads(resp.read().decode())
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = str(exc)
            time.sleep(1.0)
    raise RuntimeError(f"server not healthy within {timeout_s}s: {last_err}")


def _spawn_server(port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        [
            str(ROOT / ".venv" / "bin" / "uvicorn"),
            "app.main:app",
            "--host",
            "127.0.0.1",
            f"--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def _doc_types(docs: list[dict]) -> list[str]:
    return sorted(
        {d.get("metadata", {}).get("doc_type", "?") for d in docs}
    )


def _knowledge_built(docs: list[dict]) -> bool:
    return any(d.get("metadata", {}).get("knowledge_built") for d in docs)


def _query(
    client: httpx.Client,
    base_url: str,
    *,
    collection: str,
    query: str,
    drill_down: bool,
    top_k: int,
) -> dict[str, Any]:
    payload = {
        "collection_name": collection,
        "query": query,
        "top_k": top_k,
        "use_llm": False,
        "include_metadata": True,
        "drill_down": drill_down,
    }
    resp = client.post(f"{base_url}/query", json=payload, timeout=120.0)
    resp.raise_for_status()
    return resp.json()


def run_smoke(
    base_url: str,
    collection: str,
    probes_path: Path,
    top_k: int,
) -> dict[str, Any]:
    probes = load_probes(probes_path)
    report: dict[str, Any] = {
        "base_url": base_url,
        "collection": collection,
        "mode": "retrieval_only",
        "health": {},
        "collections": {},
        "probes": [],
        "passed": False,
    }

    with httpx.Client() as client:
        health = client.get(f"{base_url}/health", timeout=30.0)
        health.raise_for_status()
        report["health"] = health.json()

        coll_resp = client.get(f"{base_url}/collections", timeout=30.0)
        coll_resp.raise_for_status()
        report["collections"] = coll_resp.json()

        names = [
            c.get("name")
            for c in report["collections"].get("collections", [])
            if isinstance(c, dict)
        ]
        if collection not in names:
            raise RuntimeError(
                f"collection {collection!r} not in /collections: {names}"
            )

        recalls: list[float] = []
        drill_paper_hits = 0

        for probe in probes:
            flat = _query(
                client,
                base_url,
                collection=collection,
                query=probe.query,
                drill_down=False,
                top_k=top_k,
            )
            flat_docs = flat.get("retrieved_documents", [])
            flat_recall = compute_keyword_recall(
                [d.get("content", "") for d in flat_docs],
                probe.expected_keywords,
            )
            recalls.append(flat_recall)

            drill = _query(
                client,
                base_url,
                collection=collection,
                query=probe.query,
                drill_down=True,
                top_k=top_k,
            )
            drill_docs = drill.get("retrieved_documents", [])
            drill_recall = compute_keyword_recall(
                [d.get("content", "") for d in drill_docs],
                probe.expected_keywords,
            )
            paper_hits = sum(
                1
                for d in drill_docs
                if d.get("metadata", {}).get("doc_type") == "paper_summary"
            )
            drill_paper_hits += paper_hits

            report["probes"].append(
                {
                    "probe_id": probe.id,
                    "recall_flat": round(flat_recall, 4),
                    "recall_drill_down": round(drill_recall, 4),
                    "flat_doc_types": _doc_types(flat_docs),
                    "drill_doc_types": _doc_types(drill_docs),
                    "drill_paper_summary_hits": paper_hits,
                    "knowledge_built": _knowledge_built(drill_docs),
                    "processing_time_s": drill.get("processing_time_seconds"),
                }
            )

        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        report["avg_keyword_recall_flat"] = round(avg_recall, 4)
        report["drill_down_paper_hits_total"] = drill_paper_hits
        report["passed"] = (
            report["health"].get("qdrant_connected") is True
            and avg_recall > 0.50
            and drill_paper_hits > 0
            and all(p["knowledge_built"] for p in report["probes"])
        )

    return report


def run_generation_smoke(
    base_url: str,
    collection: str,
    *,
    model_profile: str = "fast",
) -> dict[str, Any]:
    """Single live query with MLX generation + drill_down (needs ~2.6 GB headroom)."""
    query = "How do language models transcribe endangered languages?"
    with httpx.Client() as client:
        t0 = time.perf_counter()
        resp = client.post(
            f"{base_url}/query",
            json={
                "collection_name": collection,
                "query": query,
                "top_k": 5,
                "use_llm": True,
                "drill_down": True,
                "include_metadata": True,
                "model_profile": model_profile,
            },
            timeout=300.0,
        )
        elapsed = round(time.perf_counter() - t0, 2)
        resp.raise_for_status()
        data = resp.json()

    answer = (data.get("answer") or "").strip()
    docs = data.get("retrieved_documents", [])
    doc_types = _doc_types(docs)
    has_citation_hint = any(
        token in answer
        for token in ("[paper:", "[topic:", "WARDEN", "language model")
    )

    return {
        "query": query,
        "model_profile": model_profile,
        "model_used": data.get("model_used"),
        "processing_time_s": data.get("processing_time_seconds", elapsed),
        "answer_chars": len(answer),
        "answer_preview": answer[:400],
        "retrieved_docs": len(docs),
        "doc_types": doc_types,
        "knowledge_built": _knowledge_built(docs),
        "has_citation_hint": has_citation_hint,
        "passed": bool(answer) and _knowledge_built(docs) and "paper_summary" in doc_types,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live API smoke for make_knowledge")
    p.add_argument("--base-url", default="", help="Running API base URL (skip spawn)")
    p.add_argument("--port", type=int, default=_DEFAULT_PORT)
    p.add_argument("--collection", default="ai_papers_knowledge")
    p.add_argument("--probes", default="data/evaluation/papers_probes.json")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--output", default="reports/knowledge/api_smoke.json")
    p.add_argument(
        "--with-llm",
        action="store_true",
        help="Also run one MLX generation query (drill_down=true, model_profile=fast)",
    )
    p.add_argument("--model-profile", default="fast")
    p.add_argument(
        "--wait-ram-sec",
        type=int,
        default=0,
        help="Poll hardware_guard up to N seconds before --with-llm (0=skip)",
    )
    return p.parse_args()


def _wait_for_ram(max_sec: int) -> bool:
    guard = ROOT / "scripts" / "hardware_guard.sh"
    deadline = time.monotonic() + max_sec
    while time.monotonic() < deadline:
        result = subprocess.run(["bash", str(guard)], capture_output=True, text=True)
        if result.returncode == 0:
            print("hardware_guard passed")
            return True
        time.sleep(10)
    print("hardware_guard still failing — attempting --with-llm anyway")
    return False


def main() -> int:
    args = parse_args()
    probes_path = Path(args.probes)
    if not probes_path.is_absolute():
        probes_path = ROOT / probes_path

    proc: subprocess.Popen[str] | None = None
    base_url = args.base_url.rstrip("/") if args.base_url else ""

    try:
        if not base_url:
            proc = _spawn_server(args.port)
            base_url = f"http://127.0.0.1:{args.port}"
            print(f"spawned uvicorn pid={proc.pid} → {base_url}")
            health = _wait_for_health(base_url, _STARTUP_TIMEOUT_S)
            print(f"health: {health.get('status')} qdrant={health.get('qdrant_connected')}")

        if args.with_llm and args.wait_ram_sec > 0:
            _wait_for_ram(args.wait_ram_sec)

        report = run_smoke(
            base_url=base_url,
            collection=args.collection,
            probes_path=probes_path,
            top_k=args.top_k,
        )

        if args.with_llm:
            print("=== generation smoke (use_llm=true, drill_down=true) ===")
            gen = run_generation_smoke(
                base_url,
                args.collection,
                model_profile=args.model_profile,
            )
            report["generation"] = gen
            report["passed"] = report["passed"] and gen["passed"]

        print(json.dumps(report, indent=2))

        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}")

        if report["passed"]:
            print("API SMOKE OK")
            return 0

        print("API SMOKE FAILED — see report above")
        return 1

    except Exception as exc:
        print(f"API SMOKE ABORT: {exc}")
        if proc and proc.stdout:
            tail = proc.stdout.read()[-4000:]
            if tail.strip():
                print("--- server log tail ---")
                print(tail)
        return 1

    finally:
        if proc is not None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())