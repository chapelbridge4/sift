

from app.config import Settings, get_settings
from app.knowledge.backend import get_knowledge_backend
from app.knowledge.config import load_profile


def test_knowledge_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.KNOWLEDGE_PROFILE == "papers"
    assert settings.KNOWLEDGE_TOPIC_SCORE_BOOST == 1.2
    assert "Qwen3-4B-Instruct-2507-Q4_K_M.gguf" in settings.KNOWLEDGE_GGUF_MODEL_PATH
    assert settings.KNOWLEDGE_OUTPUT_DIR == ""


def test_profile_retrieval_boost_matches_settings_default():
    prof = load_profile("papers")
    settings = Settings(_env_file=None)
    assert prof.retrieval.topic_score_boost == settings.KNOWLEDGE_TOPIC_SCORE_BOOST


def test_get_knowledge_backend_selects_gguf_for_papers_profile(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_GGUF_MODEL_PATH", "/tmp/knowledge-test.gguf")
    get_settings.cache_clear()
    backend = get_knowledge_backend("papers")
    assert backend.__class__.__name__ == "GGUFService"
    assert backend._model_path_override == "/tmp/knowledge-test.gguf"


def test_get_knowledge_backend_uses_profile_model_path_override(monkeypatch, tmp_path):
    model_file = tmp_path / "custom.gguf"
    model_file.write_text("fake", encoding="utf-8")

    prof = load_profile("papers")
    # KnowledgeProfile is frozen — build a modified copy via replace pattern.
    from dataclasses import replace

    from app.knowledge.config import LLMCfg

    custom_llm = LLMCfg(
        backend="gguf",
        model_path=str(model_file),
        model_id=prof.llm.model_id,
        model_hf_repo=prof.llm.model_hf_repo,
        fallback_model_id=prof.llm.fallback_model_id,
    )
    custom_prof = replace(prof, llm=custom_llm)

    backend = get_knowledge_backend(custom_prof)
    assert backend.__class__.__name__ == "GGUFService"
    assert backend._model_path_override == str(model_file)


def test_get_knowledge_backend_mlx_fallback(monkeypatch):
    from dataclasses import replace

    from app.knowledge.config import LLMCfg

    prof = load_profile("papers")
    mlx_llm = LLMCfg(
        backend="mlx",
        model_path="",
        model_id=prof.llm.model_id,
        model_hf_repo=prof.llm.model_hf_repo,
        fallback_model_id=prof.llm.fallback_model_id,
    )
    custom_prof = replace(prof, llm=mlx_llm)

    backend = get_knowledge_backend(custom_prof)
    assert backend.__class__.__name__ == "LLMService"