from dataclasses import dataclass

from app.knowledge.config import load_profile
from app.knowledge.tier0_cluster import extract_claim_spans


@dataclass
class Section:
    name: str
    text: str


@dataclass
class ParsedDoc:
    paper_id: str
    source_file: str
    sections: list[Section]


def _long_sentence(topic: str) -> str:
    return (
        f"We evaluate {topic} on three public benchmarks and report consistent improvements "
        f"over strong baselines across all metrics."
    )


def test_extract_claim_spans_prioritizes_abstract_and_respects_char_bounds():
    prof = load_profile("papers")
    abstract = " ".join([_long_sentence("retrieval-augmented generation") for _ in range(2)])
    intro = _long_sentence("unrelated baselines")
    doc = ParsedDoc(
        paper_id="p1",
        source_file="p1.pdf",
        sections=[
            Section(name="Introduction", text=intro),
            Section(name="Abstract", text=abstract),
            Section(name="Methods", text="Short."),
        ],
    )

    spans = extract_claim_spans(doc, profile=prof)

    assert len(spans) >= 1
    assert all(prof.tier0.claim_min_chars <= len(s.text) <= prof.tier0.claim_max_chars for s in spans)
    assert spans[0].section.lower() == "abstract"
    assert spans[0].paper_id == "p1"
    assert all(s.embedding_id for s in spans)


def test_extract_claim_spans_skips_sections_without_valid_spans():
    prof = load_profile("papers")
    doc = ParsedDoc(
        paper_id="p2",
        source_file="p2.pdf",
        sections=[
            Section(name="References", text="[1] Smith et al."),
            Section(
                name="Conclusion",
                text=_long_sentence("transformer scaling laws"),
            ),
        ],
    )

    spans = extract_claim_spans(doc, profile=prof)

    assert len(spans) == 1
    assert spans[0].section.lower() == "conclusion"