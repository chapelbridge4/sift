import pytest
from app.services.inference import get_inference_backend, InferenceBackend

def test_factory_returns_gguf_by_default(monkeypatch):
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
