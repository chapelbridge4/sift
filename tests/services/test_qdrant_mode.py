"""
Test that QdrantService supports embedded (zero-infra) mode via AsyncQdrantClient(path=...).
No Qdrant server is required when QDRANT_MODE=embedded.
"""

import asyncio
import os
import pytest

from app.services.qdrant_service import QdrantService


def test_embedded_mode_connects_without_server(monkeypatch, tmp_path):
    """
    QdrantService.initialize() must succeed with QDRANT_MODE=embedded
    using an on-disk path, without any running Qdrant server.

    Settings cache workaround: monkeypatch the env vars AND set the
    attribute directly on the cached Settings object so get_settings()
    lru_cache doesn't serve a stale "server" mode value.
    """
    monkeypatch.setenv("QDRANT_MODE", "embedded")
    qdrant_path = str(tmp_path / "q")
    monkeypatch.setenv("QDRANT_PATH", qdrant_path)

    svc = QdrantService()
    # Override settings attributes directly in case lru_cache returned
    # a pre-existing Settings object that hasn't seen the monkeypatched env.
    svc.settings.QDRANT_MODE = "embedded"
    svc.settings.QDRANT_PATH = qdrant_path

    asyncio.run(svc.initialize())

    assert svc.client is not None


def test_embedded_mode_idempotent(monkeypatch, tmp_path):
    """Calling initialize() twice must not raise and client stays set."""
    qdrant_path = str(tmp_path / "q2")
    monkeypatch.setenv("QDRANT_MODE", "embedded")
    monkeypatch.setenv("QDRANT_PATH", qdrant_path)

    svc = QdrantService()
    svc.settings.QDRANT_MODE = "embedded"
    svc.settings.QDRANT_PATH = qdrant_path

    asyncio.run(svc.initialize())
    asyncio.run(svc.initialize())  # second call must be a no-op

    assert svc.client is not None
