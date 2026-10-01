"""Select immutable evidence sources for committed, generated documentation.

Git roots use ordinary blobs at HEAD and reject changed working-tree bytes.
Non-Git artifact trees are explicit caller-supplied trees (used by fixture
tests), with no assertion of Git provenance. Local runs remain available to
the working-directory evidence audit rather than entering committed pages.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def committed_evidence_paths(root: Path, candidates: list[Path]) -> list[Path]:
    """Filter one bounded source inventory without reading untracked receipts."""
    root = root.resolve()
    if not (root / ".git").exists():
        return candidates
    if len(candidates) > 4096:
        raise ValueError("evidence source inventory exceeds 4096 paths")
    relative = []
    for path in candidates:
        if not path.absolute().is_relative_to(root):
            raise ValueError("evidence candidate escapes repository root")
        relative.append(path.absolute().relative_to(root).as_posix())
    if not relative:
        return []
    result = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "-r", "-z", "HEAD", "--", *relative],
        check=True,
        capture_output=True,
        timeout=30,
    )
    inventory = {}
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, name = record.split(b"\t", 1)
        mode, kind, oid_bytes = metadata.split()
        if mode in (b"100644", b"100755") and kind == b"blob":
            inventory[name.decode("utf-8")] = oid_bytes.decode("ascii")
    selected = []
    for path, rel in zip(candidates, relative, strict=True):
        oid = inventory.get(rel)
        if oid is None:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"committed evidence source is a symlink: {rel}")
        data = path.read_bytes()
        header = f"blob {len(data)}\0".encode("ascii")
        if len(oid) == 40:
            digest = hashlib.sha1(header + data, usedforsecurity=False).hexdigest()
        elif len(oid) == 64:
            digest = hashlib.sha256(header + data).hexdigest()
        else:
            raise ValueError("unsupported Git object hash format")
        if digest != oid:
            raise ValueError(f"committed evidence source differs from HEAD: {rel}")
        selected.append(path)
    return selected
