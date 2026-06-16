"""Path-containment guard for the document-ingestion boundary.

Prevents path traversal / arbitrary file read: every ingestion path must
resolve to a real file located inside the configured corpus root.
"""
from __future__ import annotations
from pathlib import Path


class UnsafePathError(ValueError):
    """Raised when an ingestion path escapes the allowed corpus root."""


def resolve_safe_paths(paths: list[str], allowed_root: str | Path) -> list[str]:
    root = Path(allowed_root).resolve()
    safe: list[str] = []
    for p in paths:
        candidate = Path(p)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()  # follows symlinks → blocks symlink escape
        if not (resolved == root or root in resolved.parents):
            raise UnsafePathError(f"path escapes corpus root: {p!r}")
        if not resolved.is_file():
            raise UnsafePathError(f"not a readable file inside corpus root: {p!r}")
        safe.append(str(resolved))
    return safe
