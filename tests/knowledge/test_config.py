import pytest

from app.knowledge.config import load_profile


def test_load_papers_profile_has_no_magic_numbers_in_code():
    prof = load_profile("papers")
    assert prof.tier0.max_clusters == 20
    assert prof.tier1.max_output_tokens == 800
    assert prof.tier2.temperature == 0.3
    assert prof.retrieval.topic_score_boost == 1.2
    assert prof.chunk.chunk_size == 512


def test_unknown_profile_fails_fast():
    with pytest.raises(FileNotFoundError):
        load_profile("does-not-exist")