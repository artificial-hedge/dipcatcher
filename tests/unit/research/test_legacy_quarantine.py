"""Quarantine registry pin: every retained artifact fails verification exactly
the way the registry records — no drifting reason, no silent "fix" that would
move a pre-contract receipt back into the verified set, and nothing under the
quarantine subtree escapes registration.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from quant_fund.research.legacy_unsealed import (
    QUARANTINE_CATEGORIES,
    classify_receipt,
    load_quarantine,
)
from quant_fund.research.receipt_v2 import verify_receipt_file

REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPTS = REPO_ROOT / "receipts"
LEGACY_DIR = RECEIPTS / "legacy-unsealed"
PIN_PATH = REPO_ROOT / "quality" / "legacy_quarantine.json"


def test_quarantine_pin_covers_exactly_the_retired_set() -> None:
    registry = load_quarantine(PIN_PATH)
    on_disk = {f"legacy-unsealed/{p.name}" for p in LEGACY_DIR.glob("*.json") if p.is_file()}
    registered = {rel for rel in registry.files if rel.startswith("legacy-unsealed/")}
    assert on_disk == registered, (
        "legacy-unsealed/ contents and the registry disagree — a retired artifact "
        "was added/removed without updating quality/legacy_quarantine.json"
    )


def test_registered_artifacts_match_their_pinned_bytes_and_reasons() -> None:
    registry = load_quarantine(PIN_PATH)
    assert registry.files, "legacy_quarantine.json must pin files"
    for rel, entry in registry.files.items():
        path = RECEIPTS / rel
        assert entry.category in QUARANTINE_CATEGORIES
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry.sha256, f"{rel}: bytes drifted from the quarantine pin"
        result = verify_receipt_file(path)
        # A quarantined artifact must never verify cleanly — if it did, it
        # belongs back in the verified set and the entry must be removed.
        assert not result["valid"], f"{rel}: quarantined artifact unexpectedly verifies"
        status, why = classify_receipt(
            rel, digest, valid=False, errors=list(result["errors"]), quarantine=registry
        )
        assert status == "quarantined", f"{rel}: {why}"


def test_schema_drift_acks_pin_existing_valid_receipts() -> None:
    """A drift acknowledgment pins every member of the drifting tag group by
    bytes, and every member is a receipt that verifies (drift is a corpus
    shape finding, never a cover for an invalid receipt)."""
    registry = load_quarantine(PIN_PATH)
    for tag, ack in registry.schema_drift_acks.items():
        assert len(ack.members) >= 2, tag
        for rel, digest in ack.members.items():
            path = RECEIPTS / rel
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"{tag}:{rel}"
            assert verify_receipt_file(path)["valid"], f"{tag}:{rel}"
            assert rel not in registry.files, f"{tag}:{rel} is also quarantined"
