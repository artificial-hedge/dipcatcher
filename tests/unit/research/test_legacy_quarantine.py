"""Quarantine registry pin: every retained artifact fails verification exactly
the way the registry records — no drifting reason, no silent "fix" that would
move a pre-contract receipt back into the verified set, and nothing under the
quarantine subtree escapes registration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from quant_fund.research.legacy_unsealed import (
    HONESTY_GAP,
    QUARANTINE_CATEGORIES,
    Quarantine,
    QuarantineEntry,
    classify_receipt,
    honesty_stamp_errors,
    load_quarantine,
    registry_base_errors,
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
        if entry.category == HONESTY_GAP:
            # The one category that verifies: it is quarantined for the
            # admission honesty stamps it lacks, re-derived from its bytes.
            assert result["valid"], f"{rel}: honesty-gap entry must verify"
        else:
            # Any other quarantined artifact must never verify cleanly — if
            # it did, it belongs back in the verified set.
            assert not result["valid"], f"{rel}: quarantined artifact unexpectedly verifies"
        status, why = classify_receipt(
            rel,
            digest,
            valid=bool(result["valid"]),
            errors=list(result["errors"]),
            quarantine=registry,
            raw=path.read_bytes(),
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


def _gap_registry(rel: str, raw: bytes, errors: tuple[str, ...]) -> Quarantine:
    entry = QuarantineEntry(
        path=rel,
        sha256=hashlib.sha256(raw).hexdigest(),
        errors=errors,
        category=HONESTY_GAP,
        reason="test fixture",
    )
    return Quarantine(files={rel: entry}, schema_drift_acks={}, registry_sha256="0" * 64)


def test_honesty_gap_entries_fail_closed() -> None:
    """``admission_honesty_gap`` holds only a receipt that verifies yet lacks
    exactly the recorded honesty stamps; every other state is invalid."""
    body = {"kind": "x", "payload": {"research_only": True, "live_pnl_claim": False}}
    raw = json.dumps(body).encode()
    sha = hashlib.sha256(raw).hexdigest()
    assert honesty_stamp_errors(body) == ([], ["data_label_missing"])
    reg = _gap_registry("r.json", raw, ("data_label_missing",))

    def classify(
        *,
        valid: bool = True,
        errors: tuple[str, ...] = (),
        sha: str = sha,
        raw: bytes | None = raw,
    ) -> tuple[str, list[str]]:
        return classify_receipt(
            "r.json", sha, valid=valid, errors=list(errors), quarantine=reg, raw=raw
        )

    assert classify() == ("quarantined", [])
    assert classify(raw=None) == ("invalid", ["quarantine_honesty_unreadable"])
    assert classify(valid=False, errors=("seal",)) == (
        "invalid",
        ["quarantine_category_mismatch", "seal"],
    )
    assert classify(sha="0" * 64) == ("invalid", ["quarantine_bytes_drift"])
    # The pin binds the classified bytes themselves, not the caller's digest.
    other = json.dumps({**body, "kind": "y"}).encode()
    assert classify(raw=other) == ("invalid", ["quarantine_bytes_drift"])

    def pinned(variant: bytes, errors: tuple[str, ...] = ("data_label_missing",)) -> list[str]:
        """Classify ``variant`` against an entry pinned to those exact bytes."""
        digest = hashlib.sha256(variant).hexdigest()
        status, why = classify_receipt(
            "r.json",
            digest,
            valid=True,
            errors=[],
            quarantine=_gap_registry("r.json", variant, errors),
            raw=variant,
        )
        assert status == "invalid"
        return why

    # The gap was closed (a label appeared): the entry is stale, not a pass.
    labelled = json.dumps({**body, "payload": {**body["payload"], "data_label": "META"}}).encode()
    assert pinned(labelled) == ["quarantine_entry_admissible"]
    # A different gap than the one recorded is drift.
    wider = json.dumps({"kind": "x", "payload": {"live_pnl_claim": False}}).encode()
    assert pinned(wider)[0] == "quarantine_errors_drift"
    # A reject-class stamp is never quarantinable.
    lying = json.dumps({"kind": "x", "payload": {"live_pnl_claim": True}}).encode()
    assert pinned(lying)[0] == "quarantine_honesty_reject"
    # Duplicate keys give one byte string two readings: unreadable.
    dup = b'{"kind": "x", "payload": {}, "payload": {"research_only": true}}'
    assert pinned(dup) == ["quarantine_honesty_unreadable"]


def _entry(rel: str, sha: str, category: str = "claim_absent") -> QuarantineEntry:
    return QuarantineEntry(
        path=rel, sha256=sha, errors=("x",), category=category, reason="test fixture"
    )


def _reg(*entries: QuarantineEntry) -> Quarantine:
    return Quarantine(files={e.path: e for e in entries}, schema_drift_acks={})


def test_registry_may_grow_only_over_base_bytes() -> None:
    old, new, stamped = "1" * 64, "2" * 64, "3" * 64
    base_files = {"a.json": old, "legacy-unsealed/moved.json": old, "s.json": stamped}
    base_stamped = {"s.json": stamped}
    empty = _reg()
    # A receipt this change adds, registered in the same change: refused.
    assert registry_base_errors(_reg(_entry("n.json", new)), empty, base_files, base_stamped) == [
        "quarantine_entry_postdates_base:'n.json'"
    ]
    # Bytes the base had: at the same path, relocated under the same name,
    # or stamped by the base head (a restoration).
    ok = _reg(_entry("a.json", old), _entry("moved.json", old), _entry("restored.json", stamped))
    assert registry_base_errors(ok, empty, base_files, base_stamped) == []
    # An entry carried over unchanged from the base registry is history.
    assert (
        registry_base_errors(
            _reg(_entry("n.json", new)), _reg(_entry("n.json", new)), base_files, base_stamped
        )
        == []
    )


def test_honesty_gap_may_only_hold_an_unadmitted_base_file() -> None:
    old, stamped = "1" * 64, "3" * 64
    base_files = {"a.json": old, "s.json": stamped, "elsewhere/b.json": old}
    base_stamped = {"s.json": stamped}
    empty = _reg()
    assert (
        registry_base_errors(
            _reg(_entry("a.json", old, HONESTY_GAP)), empty, base_files, base_stamped
        )
        == []
    )
    # Demoting a receipt the base already stamped: refused.
    assert registry_base_errors(
        _reg(_entry("s.json", stamped, HONESTY_GAP)), empty, base_files, base_stamped
    ) == ["quarantine_honesty_gap_not_unadmitted_at_base:'s.json'"]
    # Bytes that were elsewhere at base are not "this file, unadmitted".
    assert registry_base_errors(
        _reg(_entry("b.json", old, HONESTY_GAP)), empty, base_files, base_stamped
    ) == ["quarantine_honesty_gap_not_unadmitted_at_base:'b.json'"]
