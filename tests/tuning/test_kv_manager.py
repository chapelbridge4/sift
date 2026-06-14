"""
Real tests for KVManager: save/load round-trip and LRU eviction.

KVManager uses mx.savez / mx.load for MLX arrays. Tests use a temp
directory and mock those calls to avoid real MLX I/O, while exercising
the path-routing and eviction logic against the real filesystem.
"""

from unittest.mock import patch, MagicMock
import tempfile
import time
from pathlib import Path

import app.core.kv_manager as kv_mod
from app.core.kv_manager import KVManager


def _patch_kv_dir(tmp_path: Path):
    """Return a context-manager stack that redirects _DIR to tmp_path."""
    return patch.object(kv_mod, "_DIR", tmp_path)


def test_kv_manager_save_and_load():
    """Save a cache value for a session_id, then load it back."""
    fake_cache = MagicMock(name="fake_kv_cache")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # mx.savez writes a real .npz; we mock it to just touch the file so
        # the path exists without requiring a real MLX array.
        def fake_savez(path, **kwargs):
            Path(path).touch()

        # mx.load returns a dict-like; we mock it to return our fake cache.
        def fake_load(path):
            return {"cache": fake_cache}

        with (
            _patch_kv_dir(tmp),
            patch.object(kv_mod.mx, "savez", side_effect=fake_savez),
            patch.object(kv_mod.mx, "load", side_effect=fake_load),
        ):
            session_id = "test-session-abc"

            KVManager.save(session_id, fake_cache)

            # Verify the file was created under the temp dir
            expected_path = KVManager._path(session_id)
            assert expected_path.exists(), (
                f"Expected cache file at {expected_path} but it was not found."
            )

            result = KVManager.load(session_id)

        assert result is fake_cache, (
            f"load() should return the saved cache object, got {result!r}"
        )


def test_kv_manager_lru_eviction():
    """After saving N+1 sessions with max=N, the oldest file is deleted."""
    max_sessions = 3

    def fake_savez(path, **kwargs):
        Path(path).touch()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with (
            _patch_kv_dir(tmp),
            patch.object(kv_mod, "_MAX_SESSIONS", max_sessions),
            patch.object(kv_mod.mx, "savez", side_effect=fake_savez),
        ):
            session_ids = [f"session-{i}" for i in range(max_sessions + 1)]

            for sid in session_ids:
                KVManager.save(sid, object())
                # Small sleep so mtime ordering is deterministic
                time.sleep(0.02)

            remaining_files = list(tmp.glob("*.npz"))

        assert len(remaining_files) == max_sessions, (
            f"Expected {max_sessions} files after LRU eviction, "
            f"found {len(remaining_files)}: {[f.name for f in remaining_files]}"
        )

        # The oldest session's file must be gone
        with _patch_kv_dir(tmp):
            oldest_path = KVManager._path(session_ids[0])
        assert not oldest_path.exists(), (
            f"Oldest session file {oldest_path.name} should have been evicted."
        )
