"""
Real tests for KVManager: save/load round-trip and LRU eviction.

KVManager uses mlx.core.savez / mlx.core.load for MLX arrays. Tests use a
temp directory and mock those calls to avoid real MLX I/O, while exercising
the path-routing and eviction logic against the real filesystem.

mlx.core is imported lazily inside each method (so the module works on
Linux where mlx is absent). We patch at the mlx.core module level so the
lazy ``import mlx.core as mx`` inside the methods picks up the mock.
"""

import sys
import tempfile
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import app.core.kv_manager as kv_mod
from app.core.kv_manager import KVManager


def _patch_kv_dir(tmp_path: Path):
    """Return a context-manager stack that redirects _DIR to tmp_path."""
    return patch.object(kv_mod, "_DIR", tmp_path)


def _install_fake_mlx(savez_side_effect=None, load_side_effect=None):
    """
    Install a fake mlx.core into sys.modules and return (fake_mlx, fake_mx,
    orig_mlx, orig_mlx_core) so the caller can restore on teardown.

    The lazy ``import mlx.core as mx`` inside KVManager methods will resolve
    to fake_mx because sys.modules is checked before the filesystem.
    """
    fake_mx = types.ModuleType("mlx.core")
    if savez_side_effect is not None:
        fake_mx.savez = savez_side_effect
    if load_side_effect is not None:
        fake_mx.load = load_side_effect

    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_mx

    orig_mlx = sys.modules.get("mlx")
    orig_mlx_core = sys.modules.get("mlx.core")
    sys.modules["mlx"] = fake_mlx
    sys.modules["mlx.core"] = fake_mx
    return orig_mlx, orig_mlx_core


def _restore_mlx(orig_mlx, orig_mlx_core):
    if orig_mlx is None:
        sys.modules.pop("mlx", None)
    else:
        sys.modules["mlx"] = orig_mlx
    if orig_mlx_core is None:
        sys.modules.pop("mlx.core", None)
    else:
        sys.modules["mlx.core"] = orig_mlx_core


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

        orig_mlx, orig_mlx_core = _install_fake_mlx(
            savez_side_effect=fake_savez,
            load_side_effect=fake_load,
        )
        try:
            with _patch_kv_dir(tmp):
                session_id = "test-session-abc"

                KVManager.save(session_id, fake_cache)

                # Verify the file was created under the temp dir
                expected_path = KVManager._path(session_id)
                assert expected_path.exists(), (
                    f"Expected cache file at {expected_path} but it was not found."
                )

                result = KVManager.load(session_id)
        finally:
            _restore_mlx(orig_mlx, orig_mlx_core)

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

        orig_mlx, orig_mlx_core = _install_fake_mlx(savez_side_effect=fake_savez)
        try:
            with (
                _patch_kv_dir(tmp),
                patch.object(kv_mod, "_MAX_SESSIONS", max_sessions),
            ):
                session_ids = [f"session-{i}" for i in range(max_sessions + 1)]

                for sid in session_ids:
                    KVManager.save(sid, object())
                    # Small sleep so mtime ordering is deterministic
                    time.sleep(0.02)

                remaining_files = list(tmp.glob("*.npz"))
        finally:
            _restore_mlx(orig_mlx, orig_mlx_core)

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
