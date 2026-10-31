"""Legacy quarantine pin: every retired artifact fails verification exactly the
way the pin records — no drifting reason, no silent "fix" that would move a
pre-contract receipt back into the verified set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from quant_fund.research.receipt_v2 import verify_receipt_file

REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_DIR = REPO_ROOT / "receipts" / "legacy-unsealed"
PIN_PATH = REPO_ROOT / "quality" / "legacy_quarantine.json"


def _pin() -> dict[str, dict[str, object]]:
    doc = json.loads(PIN_PATH.read_text())
    assert doc.get("schema") == "legacy_quarantine.v1"
    files = doc.get("files")
    assert isinstance(files, dict) and files, "legacy_quarantine.json must pin files"
    return files


def test_quarantine_pin_covers_exactly_the_retired_set() -> None:
    on_disk = {p.name for p in LEGACY_DIR.glob("*.json") if p.is_file()}
    assert on_disk == set(_pin()), (
        "legacy-unsealed/ contents and the pin disagree — a retired artifact "
        "was added/removed without updating quality/legacy_quarantine.json"
    )


def test_retired_artifacts_match_their_pinned_bytes_and_reasons() -> None:
    for name, entry in _pin().items():
        path = LEGACY_DIR / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"{name}: bytes drifted from the quarantine pin"
        result = verify_receipt_file(path)
        # A quarantined artifact must never verify cleanly — if it did, it
        # belongs in receipts/, not here.
        assert not result["valid"], f"{name}: retired artifact unexpectedly verifies"
        assert sorted(result["errors"]) == sorted(entry["errors"]), (
            f"{name}: verify errors drifted from the recorded quarantine reason"
        )
