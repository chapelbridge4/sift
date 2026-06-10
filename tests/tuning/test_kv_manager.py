from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path


def test_kv_manager_save_and_load():
    # Mock mx.savez and mx.load
    pass


def test_kv_manager_lru_eviction():
    # After saving 4 sessions with max=3, oldest is deleted
    pass