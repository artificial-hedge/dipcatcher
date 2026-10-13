"""Evidence quarantine: retained receipts that do not verify, byte-pinned.

The receipts corpus is append-only — the epoch chain flags both deletion and
rewrite — so a receipt that fails ``verify-receipt`` is either a producer
defect to fix going forward, or historical record that cannot honestly be
re-derived. The latter stays in the corpus *quarantined*: never counted as
verified, never harvested into pooled evidence, and pinned so the quarantine
itself cannot drift. Re-sealing such a receipt, editing a missing claim into
it, or moving it out of an anchored path would forge or destroy evidence;
quarantine records it instead.

``quality/legacy_quarantine.json`` (``legacy_quarantine.v2``) is the single
registry, shared by ``suite-health --strict``, ``verify-all`` and the
committed-receipt tests so unit-test and production acceptance cannot
diverge. Keys are POSIX paths relative to the receipts dir; each entry pins

- ``sha256`` — the file's exact bytes,
- ``errors`` — the exact (sorted, de-duplicated) ``verify-receipt`` error set,
- ``category`` — one of ``QUARANTINE_CATEGORIES``,
- ``reason`` — why the bytes are retained rather than fixed.

A receipt is ``quarantined`` only when path, bytes and error set all match.
Anything else fails closed: an unregistered failure is ``invalid``; a
registered file whose bytes or errors drifted is ``invalid``; a registered
file that now *verifies* is ``invalid`` (``quarantine_entry_verifies`` — move
it back out and shrink the registry); a registry entry whose file is gone is
a stale entry. Every file under a ``QUARANTINED_SUBDIRS`` subtree must be
registered.

``schema_drift_acks`` records schema-drift groups that exist in sealed,
epoch-pinned history (two shapes under one tag). Each entry pins every member
file of the group by bytes plus a reason; ``schema_drift`` reports the group
as acknowledged drift only while the live group matches exactly. Retagging a
sealed receipt to make the drift disappear would forge it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeGuard

from quant_fund.utils.receipt import QUARANTINED_SUBDIRS, load_json_strict

QUARANTINE_SCHEMA = "legacy_quarantine.v2"
DEFAULT_QUARANTINE_PATH = Path("quality") / "legacy_quarantine.json"

#: Why a retained receipt cannot verify. Adding a category is a policy
#: change, not a data change — it needs review like any verifier change.
QUARANTINE_CATEGORIES: Mapping[str, str] = {
    "pre_seal_era": "written before the receipt_sha256 seal convention existed",
    "detached_prereg_seal": (
        "preregistration document or its prereg_seal.v1 sidecar: bound by a detached "
        "sha256 seal (optionally RFC 3161-anchored) that verify-receipt does not implement"
    ),
    "contract_postdates_receipt": (
        "sealed before its kind's deep contract required a field the bytes lack"
    ),
    "tape_manifest_unpinned": (
        "binds a non-synthetic tape digest that no committed tape_manifest.v1 attests"
    ),
    "claim_absent": (
        "sealed without a declaration the current honesty contract requires; adding it "
        "after the seal would forge the receipt"
    ),
}

ReceiptStatus = Literal["verified", "quarantined", "invalid"]

_ENTRY_FIELDS = frozenset({"sha256", "errors", "category", "reason"})
_DRIFT_ACK_FIELDS = frozenset({"members", "reason"})


def _is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class QuarantineEntry:
    path: str
    sha256: str
    errors: tuple[str, ...]
    category: str
    reason: str


@dataclass(frozen=True)
class SchemaDriftAck:
    tag: str
    members: Mapping[str, str]
    reason: str


@dataclass(frozen=True)
class Quarantine:
    files: Mapping[str, QuarantineEntry] = field(default_factory=dict)
    schema_drift_acks: Mapping[str, SchemaDriftAck] = field(default_factory=dict)
    #: sha256 of the registry bytes this view was loaded from (None = empty).
    registry_sha256: str | None = None


EMPTY_QUARANTINE = Quarantine()


def _rel_ok(rel: object) -> TypeGuard[str]:
    """A canonical relative POSIX path: no absolute/drive/ADS forms, no empty,
    ``.`` or ``..`` segments — otherwise an entry could silently never match
    the file it claims to describe (and escape the stale-entry check)."""
    if not isinstance(rel, str) or not rel or "\\" in rel or ":" in rel:
        return False
    parts = rel.split("/")
    return all(part not in ("", ".", "..") for part in parts)


def _parse_entry(rel: object, entry: object) -> QuarantineEntry:
    if not _rel_ok(rel):
        raise ValueError(f"quarantine path {rel!a} must be a relative POSIX path")
    if not isinstance(entry, Mapping) or set(entry) != _ENTRY_FIELDS:
        raise ValueError(f"quarantine entry {rel!a} must carry exactly {sorted(_ENTRY_FIELDS)}")
    errors = entry["errors"]
    if (
        not isinstance(errors, list)
        or not errors
        or not all(isinstance(e, str) and e for e in errors)
        or errors != sorted(set(errors))
    ):
        raise ValueError(f"quarantine entry {rel!a}: errors must be a sorted unique list")
    if not _is_hex(entry["sha256"], 64):
        raise ValueError(f"quarantine entry {rel!a}: sha256 must be 64 lowercase hex")
    if entry["category"] not in QUARANTINE_CATEGORIES:
        raise ValueError(f"quarantine entry {rel!a}: unknown category {entry['category']!r}")
    if not isinstance(entry["reason"], str) or not entry["reason"].strip():
        raise ValueError(f"quarantine entry {rel!a}: reason must be non-empty")
    return QuarantineEntry(
        path=rel,
        sha256=entry["sha256"],
        errors=tuple(errors),
        category=entry["category"],
        reason=entry["reason"],
    )


def _parse_drift_ack(tag: object, ack: object) -> SchemaDriftAck:
    if not isinstance(tag, str) or not tag:
        raise ValueError(f"schema drift ack tag {tag!a} must be a non-empty string")
    if not isinstance(ack, Mapping) or set(ack) != _DRIFT_ACK_FIELDS:
        raise ValueError(f"schema drift ack {tag!a} must carry exactly {sorted(_DRIFT_ACK_FIELDS)}")
    members = ack["members"]
    if (
        not isinstance(members, Mapping)
        or len(members) < 2
        or not all(_rel_ok(m) and _is_hex(d, 64) for m, d in members.items())
    ):
        raise ValueError(f"schema drift ack {tag!a}: members must map >= 2 paths to sha256")
    if not isinstance(ack["reason"], str) or not ack["reason"].strip():
        raise ValueError(f"schema drift ack {tag!a}: reason must be non-empty")
    return SchemaDriftAck(tag=tag, members=dict(members), reason=ack["reason"])


def load_quarantine(path: Path | str) -> Quarantine:
    """Parse and validate the registry; raise ``ValueError`` on any defect.

    Skipping a malformed entry could silently widen or narrow what counts as
    quarantined, so every shape violation raises instead.
    """
    raw_bytes = Path(path).read_bytes()
    doc = load_json_strict(raw_bytes)
    if not isinstance(doc, Mapping) or doc.get("schema") != QUARANTINE_SCHEMA:
        raise ValueError(f"quarantine registry must declare schema {QUARANTINE_SCHEMA!r}")
    if set(doc) - {"schema", "files", "schema_drift_acks"}:
        raise ValueError("quarantine registry carries unknown top-level keys")
    files_raw = doc.get("files")
    acks_raw = doc.get("schema_drift_acks", {})
    if not isinstance(files_raw, Mapping) or not isinstance(acks_raw, Mapping):
        raise ValueError("quarantine registry 'files'/'schema_drift_acks' must be objects")
    return Quarantine(
        files={rel: _parse_entry(rel, entry) for rel, entry in files_raw.items()},
        schema_drift_acks={tag: _parse_drift_ack(tag, a) for tag, a in acks_raw.items()},
        registry_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def load_quarantine_or_empty(path: Path | str) -> Quarantine:
    """``load_quarantine`` when the file exists; the empty registry otherwise.

    An absent registry quarantines nothing, so every failing receipt stays
    ``invalid`` — the fail-closed default for a checkout without one.
    """
    return load_quarantine(path) if Path(path).is_file() else EMPTY_QUARANTINE


def in_quarantined_subdir(rel: str) -> bool:
    parts = rel.split("/")
    return len(parts) > 1 and parts[0] in QUARANTINED_SUBDIRS


def classify_receipt(
    rel: str,
    file_sha256: str,
    *,
    valid: bool,
    errors: Sequence[str],
    quarantine: Quarantine,
) -> tuple[ReceiptStatus, list[str]]:
    """``(status, problems)`` for one receipt's ``verify-receipt`` outcome.

    ``problems`` is empty for ``verified``/``quarantined`` and names why the
    receipt is ``invalid`` otherwise (the verifier errors, plus a
    ``quarantine_*`` reason when a registry entry exists but does not match).
    """
    entry = quarantine.files.get(rel)
    if entry is None:
        if in_quarantined_subdir(rel):
            return "invalid", ["quarantine_unregistered", *errors]
        return ("verified", []) if valid else ("invalid", list(errors))
    if valid:
        return "invalid", ["quarantine_entry_verifies"]
    if file_sha256 != entry.sha256:
        return "invalid", ["quarantine_bytes_drift", *errors]
    if tuple(sorted(set(errors))) != entry.errors:
        return "invalid", ["quarantine_errors_drift", *errors]
    return "quarantined", []


def stale_quarantine_entries(root: Path, quarantine: Quarantine) -> list[str]:
    """Registry entries (quarantined files, drift-ack members) gone from ``root``."""
    out = [
        f"quarantine_entry_missing:{rel!a}"
        for rel in sorted(quarantine.files)
        if not (root / rel).is_file()
    ]
    out += [
        f"schema_drift_ack_member_missing:{tag!a}:{rel!a}"
        for tag, ack in sorted(quarantine.schema_drift_acks.items())
        for rel in sorted(ack.members)
        if not (root / rel).is_file()
    ]
    return out


def corpus_receipt_files(root: Path, *, pattern: str = "*.json") -> list[Path]:
    """Every candidate receipt under ``root``, quarantined subtrees included.

    Production verification must see the quarantine dir too — skipping it
    unverified is how a file dropped there would escape every check.
    """
    return sorted(p for p in Path(root).rglob(pattern) if p.is_file())


def quarantine_summary(
    statuses: Mapping[str, ReceiptStatus], quarantine: Quarantine
) -> dict[str, Any]:
    """Coverage accounting: how much of the corpus actually verifies."""
    n_total = len(statuses)
    n_verified = sum(1 for s in statuses.values() if s == "verified")
    n_quarantined = sum(1 for s in statuses.values() if s == "quarantined")
    by_category: dict[str, int] = {}
    for rel, status in statuses.items():
        if status == "quarantined":
            category = quarantine.files[rel].category
            by_category[category] = by_category.get(category, 0) + 1
    return {
        "registry_sha256": quarantine.registry_sha256,
        "n_registered": len(quarantine.files),
        "n_quarantined": n_quarantined,
        "by_category": dict(sorted(by_category.items())),
        "verified_fraction": (n_verified / n_total) if n_total else None,
    }
