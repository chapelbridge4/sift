import os
import tempfile

from app.config import Settings
from app.config_manager import ConfigManager


def test_config_manager_initializes_db():
    """ConfigManager creates DB and schema on first run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cm = ConfigManager(db_path=db_path)
        assert os.path.exists(db_path)
        # Check tables exist
        cur = cm.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        assert "qdrant_settings" in tables
        assert "quantization_params" in tables
        assert "hnsw_params" in tables

def test_get_default_qdrant_config():
    """Returns dict with expected Qdrant connection keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cm = ConfigManager(db_path=db_path)
        cfg = cm.get_qdrant_config()
        assert "host" in cfg
        assert "port" in cfg
        assert "grpc_port" in cfg
        assert "default_collection" in cfg

def test_detect_hardware_profile():
    """Returns 'low', 'medium', or 'high' based on RAM."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cm = ConfigManager(db_path=db_path)
        profile = cm.detect_hardware_profile()
        assert profile in ("low", "medium", "high")

def test_get_quantization_params_returns_dict():
    """Returns quantization config dict for given profile."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cm = ConfigManager(db_path=db_path)
        params = cm.get_quantization_params("medium")
        assert "enabled" in params
        assert "scalar_type" in params
        assert "always_ram" in params

def test_update_setting_persists():
    """update_setting() persists value and retrievable after close/reopen."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cm = ConfigManager(db_path=db_path)
        cm.update_setting("test_key", "test_value")
        cm.close()
        # Reopen and verify
        cm2 = ConfigManager(db_path=db_path)
        cfg = cm2.get_qdrant_config()
        assert cfg.get("test_key") == "test_value"

def test_get_hnsw_params():
    """Returns HNSW params dict for given profile."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cm = ConfigManager(db_path=db_path)
        params = cm.get_hnsw_params("default")
        assert "m" in params
        assert "ef_construct" in params
        assert "full_scan_threshold" in params
        assert params["m"] == 16

def test_fallback_to_medium_profile():
    """Returns medium profile when invalid profile passed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cm = ConfigManager(db_path=db_path)
        params = cm.get_quantization_params("nonexistent_profile")
        assert params.get("hardware_profile") == "medium"

def test_close_idempotent():
    """close() can be called multiple times without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cm = ConfigManager(db_path=db_path)
        cm.close()  # First call
        cm.close()  # Second call - should not raise

def test_context_manager():
    """Supports context manager protocol (with statement)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        with ConfigManager(db_path=db_path) as cm:
            cfg = cm.get_qdrant_config()
            assert "host" in cfg
        # After exit, should be closed


def test_cors_allow_origins_parses_comma_separated_values():
    """CORS config avoids wildcard defaults while staying easy to override."""
    settings = Settings(CORS_ALLOW_ORIGINS="http://localhost:3000, http://127.0.0.1:3000")

    assert settings.cors_allow_origins() == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
