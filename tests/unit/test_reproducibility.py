"""Worktree fingerprint stays a real digest when Git LFS would false-dirty."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_worktree_sha256


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "reproducibility@example.com")
    _git(repo, "config", "user.name", "reproducibility")
    _git(repo, "config", "commit.gpgsign", "false")


def test_worktree_fingerprint_tracks_real_bytes_not_lfs_clean_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An LFS clean filter must not turn an unmodified blob into UNKNOWN.

    ``git diff HEAD --binary`` on a non-pointer blob with ``filter=lfs`` emits
    the filtered object. After the stat cache is invalidated (``touch``, or a
    racy checkout), that capture exceeds the old 5s timeout on the large
    parquet/npy files and the fingerprint collapses to ``UNKNOWN``. Phase-1
    index verification then rejects ``git_worktree_sha256``.
    """
    lfs = shutil.which("git-lfs")
    if lfs is None:
        pytest.skip("git-lfs is required to reproduce the clean-filter false dirty")
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / ".gitattributes").write_text("payload.bin filter=lfs -text\n", encoding="utf-8")
    payload = repo / "payload.bin"
    payload.write_bytes(b"same-bytes")
    _git(repo, "add", "--", ".gitattributes")
    # Store the raw bytes. A normal ``git add`` would run the LFS clean filter
    # and persist a pointer, which is not what this checkout contains.
    _git(
        repo,
        "-c",
        "filter.lfs.clean=",
        "-c",
        "filter.lfs.smudge=",
        "-c",
        "filter.lfs.process=",
        "-c",
        "filter.lfs.required=false",
        "add",
        "--",
        "payload.bin",
    )
    _git(repo, "commit", "-qm", "init")
    _git(repo, "config", "filter.lfs.process", "git-lfs filter-process")
    _git(repo, "config", "filter.lfs.required", "true")
    # Matching stat information skips the filter. Touching forces the recheck
    # that a same-second checkout leaves racy.
    payload.touch()
    false_dirty = subprocess.run(
        ["git", "diff", "HEAD", "--binary"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    assert false_dirty.stdout, "LFS clean filter did not false-dirty the blob"

    monkeypatch.chdir(repo)
    fingerprint = git_worktree_sha256()
    assert fingerprint == hash_bytes(b"")
    assert len(fingerprint) == 64

    payload.write_bytes(b"same-bytes!")
    changed = git_worktree_sha256()
    assert changed != fingerprint
    assert len(changed) == 64
    assert all(char in "0123456789abcdef" for char in changed)
