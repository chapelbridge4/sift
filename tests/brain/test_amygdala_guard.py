"""
Tests for Amygdala._load_cross_encoder memory guard.

The guard must skip the cross-encoder when available RAM is below
RERANK_MIN_AVAILABLE_GB, and load it when RAM is sufficient.
"""

from unittest.mock import MagicMock, patch


def _make_vmem(available_gb: float):
    """Return a mock svmem-like object with only .available set."""
    vmem = MagicMock()
    vmem.available = int(available_gb * 1024 ** 3)
    return vmem


def test_cross_encoder_loads_when_ram_is_high():
    """When available RAM is 4 GB (> 1.5 GB default), cross-encoder must load."""
    mock_ce_instance = MagicMock()
    mock_ce_class = MagicMock(return_value=mock_ce_instance)

    with (
        patch("app.brain.amygdala.psutil.virtual_memory", return_value=_make_vmem(4.0)),
        patch("app.brain.amygdala.TRANSFORMERS_AVAILABLE", True),
        patch("app.brain.amygdala.CrossEncoder", mock_ce_class),
    ):
        from app.brain.amygdala import Amygdala

        amygdala = Amygdala()
        result = amygdala._load_cross_encoder()

    assert result is mock_ce_instance, (
        f"Expected cross-encoder instance, got {result!r}"
    )
    mock_ce_class.assert_called_once()


def test_cross_encoder_skipped_when_ram_is_low():
    """When available RAM is 0.5 GB (< 1.5 GB default), cross-encoder must be skipped."""
    mock_ce_class = MagicMock()

    with (
        patch("app.brain.amygdala.psutil.virtual_memory", return_value=_make_vmem(0.5)),
        patch("app.brain.amygdala.TRANSFORMERS_AVAILABLE", True),
        patch("app.brain.amygdala.CrossEncoder", mock_ce_class),
    ):
        from app.brain.amygdala import Amygdala

        amygdala = Amygdala()
        result = amygdala._load_cross_encoder()

    assert result is None, (
        f"Expected None when RAM is low, got {result!r}"
    )
    mock_ce_class.assert_not_called()


def test_cross_encoder_respects_custom_min_gb():
    """RERANK_MIN_AVAILABLE_GB setting is honoured: 3 GB available < 4 GB threshold → skip."""
    mock_ce_class = MagicMock()

    with (
        patch("app.brain.amygdala.psutil.virtual_memory", return_value=_make_vmem(3.0)),
        patch("app.brain.amygdala.TRANSFORMERS_AVAILABLE", True),
        patch("app.brain.amygdala.CrossEncoder", mock_ce_class),
    ):
        from app.brain.amygdala import Amygdala

        amygdala = Amygdala()
        # Override the setting on the instance directly
        amygdala.settings = MagicMock()
        amygdala.settings.RERANK_MIN_AVAILABLE_GB = 4.0
        amygdala.settings.RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        result = amygdala._load_cross_encoder()

    assert result is None, (
        f"Expected None when available (3GB) < min_gb (4GB), got {result!r}"
    )
    mock_ce_class.assert_not_called()
