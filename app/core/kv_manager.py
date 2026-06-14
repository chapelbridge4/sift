"""Disk-persistent KV cache with LRU eviction."""

import threading
from pathlib import Path
from app.config import get_settings

_settings = get_settings()
_DIR = Path(_settings.KV_CACHE_DIR)
_MAX_SESSIONS = int(_settings.KV_MAX_SESSIONS)
_MAX_MB = int(_settings.KV_MAX_MB_PER_FILE)
_LOCK = threading.Lock()


class KVManager:
    """Manages KV cache persistence to disk with LRU eviction and size caps."""

    @staticmethod
    def _path(session_id: str) -> Path:
        from hashlib import sha256

        h = sha256(session_id.encode()).hexdigest()[:16]
        return _DIR / f"{h}.npz"

    @classmethod
    def save(cls, session_id: str, cache) -> None:
        """
        Save KV cache to disk for a session.

        Args:
            session_id: Unique session identifier
            cache: KV cache object to persist
        """
        _DIR.mkdir(parents=True, exist_ok=True)
        p = cls._path(session_id)
        with _LOCK:
            try:
                import mlx.core as mx
                mx.savez(str(p), cache=cache)
            except Exception as e:
                print(f"KV save failed: {e}")
                return

            # LRU eviction: keep only _MAX_SESSIONS newest files
            files = sorted(
                _DIR.glob("*.npz"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            for old in files[_MAX_SESSIONS:]:
                old.unlink(missing_ok=True)

            # Size cap: delete files exceeding _MAX_MB
            for f in files[:_MAX_SESSIONS]:
                if f.stat().st_size > _MAX_MB * 1024 * 1024:
                    f.unlink(missing_ok=True)

    @classmethod
    def load(cls, session_id: str):
        """
        Load KV cache from disk for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            KV cache object or None if not found / load fails
        """
        p = cls._path(session_id)
        if not p.exists():
            return None
        with _LOCK:
            try:
                import mlx.core as mx
                data = mx.load(str(p))
                return data.get("cache") if hasattr(data, "get") else data
            except Exception:
                return None

    @classmethod
    def evict(cls, session_id: str) -> None:
        """
        Evict (delete) KV cache file for a session.

        Args:
            session_id: Unique session identifier
        """
        cls._path(session_id).unlink(missing_ok=True)

    @classmethod
    def clear_all(cls) -> None:
        """Clear all KV cache files from disk."""
        with _LOCK:
            for f in _DIR.glob("*.npz"):
                f.unlink(missing_ok=True)