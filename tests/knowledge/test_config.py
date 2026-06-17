import pytest

from app.knowledge.config import load_profile


def test_load_papers_profile_has_no_magic_numbers_in_code():
    prof = load_profile("papers")
    assert prof.tier0.max_clusters == 20
    assert prof.tier0.max_sentences_per_claim == 3
    assert prof.tier1.max_output_tokens == 800
    assert prof.tier2.temperature == 0.3
    assert prof.tier2.max_claims_per_paper == 5
    assert prof.llm.max_retries == 2
    assert prof.llm.retry_backoff_base_seconds == 0.5
    assert prof.retrieval.topic_score_boost == 1.2
    assert prof.retrieval.drill_down_top_k == 5
    assert prof.chunk.chunk_size == 512
    assert "pdf" in prof.parse.extensions


def test_unknown_profile_fails_fast():
    with pytest.raises(FileNotFoundError):
        load_profile("does-not-exist")