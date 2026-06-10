import pytest
from unittest.mock import MagicMock


def test_generate_uses_skip_special_tokens():
    # Verify that whatever generates text calls decode with skip_special_tokens=True
    pass


def test_no_special_token_leak_in_output():
    # Mock output ids that include <|text|> and verify it is stripped
    pass