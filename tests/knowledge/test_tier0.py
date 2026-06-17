from unittest.mock import MagicMock

from app.knowledge.config import load_profile
from app.knowledge.models import ClaimSpan
from app.knowledge.tier0_cluster import cluster_spans


def test_cluster_groups_similar_spans():
    # two tight groups in embedding space
    embs = {
        "e1": [1.0, 0.0],
        "e2": [0.99, 0.01],
        "e3": [0.0, 1.0],
        "e4": [0.01, 0.99],
    }
    spans = [
        ClaimSpan(paper_id="p1", text="rag a", section="S", embedding_id="e1"),
        ClaimSpan(paper_id="p2", text="rag b", section="S", embedding_id="e2"),
        ClaimSpan(paper_id="p1", text="quant a", section="S", embedding_id="e3"),
        ClaimSpan(paper_id="p2", text="quant b", section="S", embedding_id="e4"),
    ]
    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[embs[s.embedding_id] for s in spans])
    prof = load_profile("papers")
    manifest = cluster_spans(spans, embedder=embedder, profile=prof)
    # spans 0,1 share a cluster; 2,3 share another
    cid = {s.embedding_id: c for c, members in manifest.clusters.items() for s in members}
    assert cid["e1"] == cid["e2"]
    assert cid["e3"] == cid["e4"]
    assert cid["e1"] != cid["e3"]