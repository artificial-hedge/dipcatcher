"""Guard: the sftp-clobber stray nested trees stay ignored and untracked.

Root cause is documented in ``tools/guard-config.ps1``: the off-machine Mac
sync runs ``put -r src /D:/dipcatcher/src`` and nests whole trees
(``src/src``, ``tests/tests``); a ``scripts/scripts`` tree was committed the
same way. Those nested trees must never be re-tracked, so they are ignored in
``.gitignore`` and this test fails if either invariant regresses.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

STRAY_TREES = ("src/src/", "tests/tests/", "scripts/scripts/")
IGNORE_PATTERNS = tuple(f"/{tree}" for tree in STRAY_TREES)


def test_nested_stray_trees_are_gitignored() -> None:
    entries = {
        line.strip() for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    missing = [pattern for pattern in IGNORE_PATTERNS if pattern not in entries]
    assert missing == [], f"nested stray trees not ignored in .gitignore: {missing}"


def test_no_tracked_files_under_nested_stray_trees() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--", *STRAY_TREES],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return  # not a git checkout (e.g. an sdist); nothing to assert
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    assert tracked == [], f"nested stray trees are git-tracked: {tracked}"
