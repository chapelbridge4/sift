from app.knowledge.artifacts import read_artifact, write_artifact
from app.knowledge.models import PaperSummary, TopicSheet


def test_paper_summary_roundtrip(tmp_path):
    p = PaperSummary(
        paper_id="2301.1",
        title="T",
        authors=["A"],
        source_file="papers/t.pdf",
        topics=["rag"],
        claims=[{"text": "c", "section": "M"}],
        methods="m",
    )
    path = write_artifact(p, tmp_path)
    assert path.exists() and path.suffix == ".md"
    back = read_artifact(path)
    assert back.doc_type == "paper_summary"
    assert back.paper_id == "2301.1"
    assert back.claims[0].text == "c"


def test_topic_sheet_roundtrip(tmp_path):
    t = TopicSheet(topic_id="rag", slug="rag", title="RAG", body="x", links_to=["2301.1"])
    path = write_artifact(t, tmp_path)
    back = read_artifact(path)
    assert back.doc_type == "topic" and back.links_to == ["2301.1"]


def test_degraded_included_in_frontmatter_when_true(tmp_path):
    t = TopicSheet(
        topic_id="rag",
        slug="rag",
        title="RAG",
        body="x",
        links_to=["2301.1"],
        degraded=True,
    )
    path = write_artifact(t, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "degraded: true" in text.lower()
    back = read_artifact(path)
    assert back.degraded is True