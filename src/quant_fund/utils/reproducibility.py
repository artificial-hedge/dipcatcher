"""Shared reproducibility fingerprints for research receipts and gates."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from quant_fund.utils.hashing import hash_bytes

# Files under ``filter=lfs`` are stored in this repo as ordinary blobs, not
# Git LFS pointers. ``git-lfs filter-process`` still rewrites them to pointers
# during ``git diff``. ``git diff HEAD --binary`` then emits the whole object
# (tens or hundreds of MB). Capturing that with a 5s timeout raises
# ``TimeoutExpired``, which this helper used to collapse to ``UNKNOWN``.
# A fresh ``actions/checkout`` often writes the worktree and the index in the
# same second, so the stat cache is racy and Git rechecks those files. Blank
# the LFS filter for this comparison and hash ``--raw`` blob ids instead of
# the binary patch: byte-identical files stay clean, and a real edit still
# changes the digest without materializing the object.
_LFS_FILTER_OFF = (
    "-c",
    "filter.lfs.clean=",
    "-c",
    "filter.lfs.smudge=",
    "-c",
    "filter.lfs.process=",
    "-c",
    "filter.lfs.required=false",
)


def git_worktree_sha256() -> str:
    """Hash tracked and relevant untracked changes in the current checkout."""
    git = shutil.which("git")
    if git is None:
        return "UNKNOWN"
    try:
        diff = subprocess.run(
            [git, *_LFS_FILTER_OFF, "diff", "--raw", "-z", "--no-renames", "HEAD"],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        untracked = subprocess.run(
            [git, "ls-files", "--others", "--exclude-standard", "-z"],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        ignored_prefixes = (
            ".venv/",
            ".pytest_cache/",
            "data/metadata/research/",
            "data/metadata/paper/",
            "htmlcov/",
        )
        additions = bytearray()
        for raw_path in untracked.split(b"\0"):
            if not raw_path:
                continue
            path = raw_path.decode("utf-8")
            if path.startswith(ignored_prefixes):
                continue
            file_path = Path(path)
            if file_path.is_file():
                additions.extend(raw_path)
                additions.extend(b"\0")
                additions.extend(file_path.read_bytes())
        return hash_bytes(diff + bytes(additions))
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"


def git_revision() -> str:
    """Return the checked-out revision, or an explicit unknown marker."""
    git = shutil.which("git")
    if git is None:
        return "UNKNOWN"
    try:
        return subprocess.run(
            [git, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"
