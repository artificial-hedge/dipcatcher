"""Generated committed pages select immutable HEAD evidence, not local runs."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from scripts._committed_evidence import committed_evidence_paths
from scripts.build_evidence_report import receipt_paths
from scripts.research_findings import collect_batches


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _repository(root: Path) -> Path:
    _git(root, "init", "-q")
    directory = root / "receipts"
    directory.mkdir()
    path = directory / "committed.json"
    path.write_text('{"schema":"probe.v1","research_only":true}\n')
    _git(root, "add", "receipts")
    _git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return path


def test_head_only_paths_exclude_local_staged_and_committed_symlink(tmp_path: Path) -> None:
    committed = _repository(tmp_path)
    local = committed.with_name("local.json")
    local.write_bytes(committed.read_bytes())
    staged = committed.with_name("staged.json")
    staged.write_bytes(committed.read_bytes())
    link = committed.with_name("link.json")
    link.symlink_to(committed.name)
    _git(tmp_path, "add", "receipts/link.json")
    _git(
        tmp_path,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "symlink",
    )
    _git(tmp_path, "add", "receipts/staged.json")
    assert committed_evidence_paths(tmp_path, [local, staged, link, committed]) == [committed]
    assert receipt_paths(tmp_path) == [committed]
    _, notes = collect_batches(tmp_path)
    assert any("committed.json" in note for note in notes)
    assert not any(
        "local.json" in note or "staged.json" in note or "link.json" in note for note in notes
    )


def test_modified_committed_evidence_fails_closed(tmp_path: Path) -> None:
    committed = _repository(tmp_path)
    committed.write_text('{"forged":true}\n')
    with pytest.raises(ValueError, match="differs from HEAD"):
        committed_evidence_paths(tmp_path, [committed])
    with pytest.raises(ValueError, match="differs from HEAD"):
        receipt_paths(tmp_path)
    with pytest.raises(ValueError, match="differs from HEAD"):
        collect_batches(tmp_path)


def test_replaced_tracked_blob_symlink_fails_closed(tmp_path: Path) -> None:
    committed = _repository(tmp_path)
    original = committed.read_bytes()
    target = committed.with_name("local.json")
    target.write_bytes(original)
    committed.unlink()
    committed.symlink_to(target.name)
    with pytest.raises(ValueError, match="symlink"):
        receipt_paths(tmp_path)


def test_non_git_fixture_tree_has_no_git_provenance_requirement(tmp_path: Path) -> None:
    path = tmp_path / "supplied.json"
    path.write_text("{}")
    assert committed_evidence_paths(tmp_path, [path]) == [path]
