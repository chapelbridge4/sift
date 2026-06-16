import pytest

from app.security import UnsafePathError, resolve_safe_paths


def test_accepts_path_inside_root(tmp_path):
    root = tmp_path / "corpus"; root.mkdir()
    f = root / "doc.txt"; f.write_text("hi")
    out = resolve_safe_paths([str(f)], root)
    assert out == [str(f.resolve())]

def test_accepts_relative_path_inside_root(tmp_path):
    root = tmp_path / "corpus"; root.mkdir()
    (root / "a.txt").write_text("x")
    out = resolve_safe_paths(["a.txt"], root)
    assert out[0] == str((root / "a.txt").resolve())

def test_rejects_absolute_outside_root(tmp_path):
    root = tmp_path / "corpus"; root.mkdir()
    with pytest.raises(UnsafePathError):
        resolve_safe_paths(["/etc/passwd"], root)

def test_rejects_dotdot_traversal(tmp_path):
    root = tmp_path / "corpus"; root.mkdir()
    with pytest.raises(UnsafePathError):
        resolve_safe_paths(["../../etc/passwd"], root)

def test_rejects_symlink_escape(tmp_path):
    root = tmp_path / "corpus"; root.mkdir()
    secret = tmp_path / "secret.txt"; secret.write_text("s")
    link = root / "link.txt"; link.symlink_to(secret)
    with pytest.raises(UnsafePathError):
        resolve_safe_paths([str(link)], root)

def test_rejects_missing_file(tmp_path):
    root = tmp_path / "corpus"; root.mkdir()
    with pytest.raises(UnsafePathError):
        resolve_safe_paths(["nope.txt"], root)

def test_empty_list_returns_empty():
    assert resolve_safe_paths([], "/tmp") == []

def test_rejects_empty_string_element(tmp_path):
    root = tmp_path / "corpus"; root.mkdir()
    with pytest.raises(UnsafePathError):
        resolve_safe_paths([""], root)
