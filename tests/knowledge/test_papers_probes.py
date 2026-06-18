import json
from pathlib import Path

from app.kbforge.eval.probe_evaluator import load_probes


def test_papers_probes_fixture_loads():
    path = Path(__file__).resolve().parents[2] / "data/evaluation/papers_probes.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    probes = load_probes(path)
    assert len(probes) >= 6
    assert data["queries"][0]["expected_keywords"]