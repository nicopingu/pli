"""PLI (10/11/2026): the code that computes a checkpoint enters its file key (pli/code_key.py)."""

from __future__ import annotations

import shutil
from pathlib import Path

from pli.additions import code_key


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "pli"
    for rel, text in {
        "additions/a1_lsa.py": "A = 1\n", "additions/ane/ffn.py": "F = 1\n", "additions/ane/native/pli_ane.mm": "// ane\n",
        "additions/events.py": "E = 1\n", "additions/trace.py": "T = 1\n", "additions/falcon/capture.py": "C = 1\n",
        "additions/extended-list-2048.txt": "1 2 3\n", "additions/__pycache__/a1_lsa.cpython-312.pyc": "x",
        "engine/family_prefill.py": "P = 1\n", "engine/grammar.py": "G = 1\n",
    }.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def _key(root: Path) -> str:
    code_key.digest.cache_clear()
    return code_key.suffix(True, root)


def test_hashed_files(tmp_path):
    root = _tree(tmp_path)
    names = [p.relative_to(root).as_posix() for p in code_key.files(root)]
    assert names == ["additions/a1_lsa.py", "additions/ane/ffn.py", "additions/ane/native/pli_ane.mm", "engine/family_prefill.py"]


def test_compute_change_changes_key(tmp_path):
    root = _tree(tmp_path)
    before = _key(root)
    assert before.startswith("|code=") and len(before) == len("|code=") + 12
    (root / "additions/a1_lsa.py").write_text("A = 2\n")
    assert _key(root) != before
    (root / "additions/a1_lsa.py").write_text("A = 1\n")
    assert _key(root) == before
    (root / "engine/family_prefill.py").write_text("P = 2\n")
    assert _key(root) != before


def test_non_compute_change_keeps_key(tmp_path):
    root = _tree(tmp_path)
    before = _key(root)
    for rel in ("additions/events.py", "additions/trace.py", "additions/falcon/capture.py", "additions/extended-list-2048.txt", "engine/grammar.py"):
        (root / rel).write_text("changed\n")
    assert _key(root) == before


def test_moved_file_changes_key(tmp_path):
    root = _tree(tmp_path)
    before = _key(root)
    shutil.move(root / "additions/ane/ffn.py", root / "additions/ffn.py")
    assert _key(root) != before


def test_off_is_previous_key():
    assert code_key.suffix(False) == ""


def test_real_tree_is_stable():
    code_key.digest.cache_clear()
    first = code_key.suffix(True)
    code_key.digest.cache_clear()
    assert code_key.suffix(True) == first
    assert any(p.name == "checkpoint_bits.py" for p in code_key.files())
