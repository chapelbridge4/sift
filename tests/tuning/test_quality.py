import pytest

pytest.importorskip("app.tuning.quality", reason="app.tuning.quality not yet implemented")


def test_detect_garbage_flags_special_tokens():
    from app.tuning.quality import detect_garbage
    assert detect_garbage("Hello <|text|> world") is True


def test_detect_garbage_allows_clean_text():
    from app.tuning.quality import detect_garbage
    assert detect_garbage("Transformer models use attention.") is False


def test_is_valid_response_requires_multiple_sentences():
    from app.tuning.quality import is_valid_response
    valid, _ = is_valid_response("The model processes input efficiently. It generates coherent responses.")
    assert valid is True


def test_is_valid_response_rejects_empty():
    from app.tuning.quality import is_valid_response
    valid, reason = is_valid_response("")
    assert valid is False
    assert reason == "too_short"