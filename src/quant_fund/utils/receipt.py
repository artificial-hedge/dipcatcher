"""Shared receipt-seal primitive.

One convention repo-wide: ``receipt_sha256`` binds the canonical JSON bytes of
the payload, computed with a small exclusion set for keys attached post-seal
(the digest itself and any filesystem-path metadata). Lane packages wrap this
with their own exclusion set and honesty contract.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

ALWAYS_EXCLUDED = ("receipt_sha256",)

# Subdirectories under a receipts root whose contents are governed by a
# separate authority (``quality/legacy_quarantine.json`` byte-pins
# ``receipts/legacy-unsealed/``) rather than per-file seal verification.
# Corpus scanners (evidence audit, suite health, lattice, receipts-reverify)
# skip these; the epoch chain still hashes them as members.
QUARANTINED_SUBDIRS = frozenset({"legacy-unsealed"})


def discover_repo_root(*starts: Path | str | None) -> Path | None:
    """Walk up from *starts* looking for a dipcatcher checkout."""
    seen: set[Path] = set()
    for start in starts:
        if start is None:
            continue
        cur = Path(start).resolve()
        if cur.is_file():
            cur = cur.parent
        for candidate in (cur, *cur.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "pyproject.toml").is_file() and (
                candidate / "quality" / "legacy_quarantine.json"
            ).is_file():
                return candidate
    return None


def _in_real_checkout(path: Path, repo_root: Path | None) -> bool:
    """True when *path* sits inside a checkout that has crown-jewel pins."""
    if repo_root is None:
        return False
    if not (repo_root / "quality" / "crown_jewels.json").is_file():
        return False
    try:
        path.resolve().relative_to(repo_root.resolve())
        return True
    except ValueError:
        return False


def _quarantine_blocks(path: Path, rel: str, file_sha256: str, repo_root: Path | None) -> bool:
    if repo_root is None:
        return False
    from quant_fund.research.legacy_unsealed import load_quarantine_or_empty

    registry = load_quarantine_or_empty(repo_root / "quality" / "legacy_quarantine.json")
    if rel in registry.files:
        return True
    if not _in_real_checkout(path, repo_root):
        return False
    return any(entry.sha256 == file_sha256 for entry in registry.files.values())


def _verifier_rejected_real_receipt(path: Path, repo_root: Path | None) -> bool:
    """Production verifier, only on real ``receipts/`` members of a checkout.

    Synthetic tmp_path stubs used by unit tests are not under the pinned
    receipts tree and must remain loadable as admitted examples.
    """
    if not _in_real_checkout(path, repo_root) or repo_root is None:
        return False
    receipts_home = repo_root / "receipts"
    if not receipts_home.is_dir():
        return False
    try:
        path.resolve().relative_to(receipts_home.resolve())
    except ValueError:
        return False
    from quant_fund.schemas.receipt_v2_leaf import verify_receipt_file

    return not verify_receipt_file(path)["valid"]


def corpus_training_refused(
    path: Path,
    receipts_root: Path,
    payload: Mapping[str, Any],
    file_sha256: str,
    *,
    repo_root: Path | None = None,
) -> bool:
    """True when a parseable receipt must never enter fx-1 corpus loading."""
    from quant_fund.research.legacy_unsealed import (
        honesty_stamp_errors,
        in_quarantined_subdir,
    )

    reject, _gaps = honesty_stamp_errors(payload)
    if reject:
        return True
    try:
        rel = path.relative_to(receipts_root).as_posix()
    except ValueError:
        return True
    if in_quarantined_subdir(rel) or any(part in QUARANTINED_SUBDIRS for part in Path(rel).parts):
        return True
    if _quarantine_blocks(path, rel, file_sha256, repo_root):
        return True
    return _verifier_rejected_real_receipt(path, repo_root)


def is_quarantined(path: Path, root: Path) -> bool:
    """True iff ``path`` sits under a quarantined top-level subdir of ``root``."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    return len(rel.parts) > 1 and rel.parts[0] in QUARANTINED_SUBDIRS


def verified_corpus_files(root: Path, *, pattern: str = "*.json") -> list[Path]:
    """Recursive corpus listing minus quarantined subdirs, sorted.

    Per-file verification must match the epoch chain's ``rglob`` member
    semantics — a claim receipt dropped in a subdirectory is still corpus
    evidence, not a blind spot."""
    return sorted(p for p in root.rglob(pattern) if p.is_file() and not is_quarantined(p, root))


def seal_receipt(receipt: Mapping[str, Any], *, exclude: Iterable[str] = ()) -> dict[str, Any]:
    """Copy of *receipt* with ``receipt_sha256`` bound over canonical JSON."""
    dropped = set(exclude) | set(ALWAYS_EXCLUDED)
    unsigned = {key: value for key, value in receipt.items() if key not in dropped}
    return {**receipt, "receipt_sha256": hash_bytes(canonical_json_bytes(unsigned))}


def seal_errors(receipt: Mapping[str, Any], *, exclude: Iterable[str] = ()) -> list[str]:
    """Digest errors for a sealed receipt; ``[]`` when the seal matches."""
    dropped = set(exclude) | set(ALWAYS_EXCLUDED)
    unsigned = {key: value for key, value in receipt.items() if key not in dropped}
    expected = hash_bytes(canonical_json_bytes(unsigned))
    return [] if receipt.get("receipt_sha256") == expected else ["receipt_sha256"]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate_json_key:{key!a}")
        obj[key] = value
    return obj


def load_json_strict(data: bytes | str) -> Any:
    """``json.loads`` that rejects duplicate object keys (last-wins would let
    one byte string carry two readings — fatal for policy files that decide
    what a gate accepts). Raises ``ValueError``."""
    return json.loads(data, object_pairs_hook=_reject_duplicate_keys)
