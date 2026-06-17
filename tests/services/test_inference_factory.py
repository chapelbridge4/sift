import pytest

from app.config import Settings
from app.services.inference import get_inference_backend


def test_inference_backend_default_is_mlx():
    """MLX is the honest default: it is the backend the profile-based RAG path
    actually uses. A fresh Settings (no env override) must report 'mlx'."""
    assert Settings(_env_file=None).INFERENCE_BACKEND == "mlx"

def test_factory_returns_gguf_when_configured(monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "gguf")
    backend = get_inference_backend()
    assert backend.__class__.__name__ == "GGUFService"
    assert hasattr(backend, "generate_rag_response")

def test_factory_returns_mlx_when_configured(monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "mlx")
    backend = get_inference_backend()
    assert backend.__class__.__name__ == "LLMService"

def test_factory_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "bogus")
    with pytest.raises(ValueError):
        get_inference_backend()
