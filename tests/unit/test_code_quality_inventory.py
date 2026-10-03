from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "code_quality_inventory", ROOT / "scripts" / "code_quality_inventory.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_measure_file_separates_source_comments_and_docstrings(tmp_path: Path) -> None:
    path = tmp_path / "sample.py"
    path.write_text(
        '"""module\n'
        'documentation"""\n'
        "\n"
        "# explanation\n"
        "class Example:\n"
        '    """class docs"""\n'
        "\n"
        "    def test_behavior(self):\n"
        '        """function docs"""\n'
        "        value = 1  # inline\n"
        "        return value\n"
    )

    measured = MODULE.measure_file(path, repo=tmp_path)

    assert measured.physical_lines == 11
    assert measured.source_lines == 4
    assert measured.comment_lines == 2
    assert measured.docstring_lines == 4
    assert measured.functions == 1
    assert measured.classes == 1
    assert measured.test_functions == 1


def test_tracked_files_ignore_untracked_and_generated_paths(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    source = tmp_path / "src" / "kept.py"
    source.parent.mkdir()
    source.write_text("answer = 42\n")
    ignored = tmp_path / "src" / "node_modules" / "ignored.py"
    ignored.parent.mkdir()
    ignored.write_text("padding = True\n")
    untracked = tmp_path / "tests" / "untracked.py"
    untracked.parent.mkdir()
    untracked.write_text("def test_nope(): pass\n")
    subprocess.run(["git", "add", str(source), str(ignored)], cwd=tmp_path, check=True)

    assert MODULE.tracked_python_files(tmp_path) == [source]


def test_inventory_minimum_is_explicit_and_fail_closed(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("def feature():\n    return 1\n")
    subprocess.run(["git", "add", str(source)], cwd=tmp_path, check=True)

    passing = MODULE.build_inventory(tmp_path, minimum_source_lines=2)
    failing = MODULE.build_inventory(tmp_path, minimum_source_lines=3)

    assert passing.meets_minimum is True
    assert failing.meets_minimum is False
    with pytest.raises(ValueError, match="non-negative"):
        MODULE.build_inventory(tmp_path, minimum_source_lines=-1)
