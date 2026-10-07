"""Hash-sealed (sidecar) forward pre-registration + tamper-detection tests.

Proves that silent edits to a sealed pre-registration are detected, both via
the embedded self-seal and the ``.seal.json`` sidecar. Also verifies the
committed machine-readable forward pre-registration is correctly sealed.
"""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund.data.prereg_seal import (
    content_hash,
    seal_file,
    seal_payload,
    verify_payload,
    verify_seal,
)

RECEIPTS = Path(__file__).resolve().parents[3] / "receipts"
COMMITTED = RECEIPTS / "forward_record_preregistration_v1.json"


def _prereg() -> dict:
    return {
        "schema": "forward_record_preregistration.v1",
        "declaration_date": "2026-10-07",
        "status": "NOT_YET_COLLECTED",
        "window": {"start": "2026-07-01", "label": "forward_2026H2"},
        "effect_bps": 5,
        "holdout_2025": "SPENT",
    }


# -- embedded self-seal --


def test_self_seal_roundtrip() -> None:
    sealed = seal_payload(_prereg())
    assert verify_payload(sealed)["valid"] is True


def test_self_seal_detects_content_tamper() -> None:
    sealed = seal_payload(_prereg())
    tampered = dict(sealed)
    tampered["effect_bps"] = 500  # silent edit after sealing
    verdict = verify_payload(tampered)
    assert verdict["valid"] is False
    assert "seal_content_hash_mismatch" in verdict["errors"]


def test_self_seal_detects_window_tamper() -> None:
    sealed = seal_payload(_prereg())
    tampered = json.loads(json.dumps(sealed))
    tampered["window"]["start"] = "2027-01-01"  # silently extend the window
    assert verify_payload(tampered)["valid"] is False


def test_content_hash_ignores_seal_block() -> None:
    sealed = seal_payload(_prereg())
    assert content_hash(sealed) == sealed["_seal"]["content_sha256"]


# -- sidecar seal --


def test_sidecar_seal_roundtrip(tmp_path) -> None:
    p = tmp_path / "prereg.json"
    p.write_text(json.dumps(seal_payload(_prereg())), encoding="utf-8")
    seal = seal_file(p)
    assert verify_seal(p)["valid"] is True
    assert seal["sha256"] == verify_seal(p)["actual"]


def test_sidecar_seal_detects_file_tamper(tmp_path) -> None:
    p = tmp_path / "prereg.json"
    p.write_text(json.dumps(seal_payload(_prereg())), encoding="utf-8")
    seal_file(p)
    p.write_text(json.dumps(seal_payload(_prereg() | {"effect_bps": 999})), encoding="utf-8")
    verdict = verify_seal(p)
    assert verdict["valid"] is False
    assert "seal_sha256_mismatch" in verdict["errors"]


def test_sidecar_seal_missing_is_fail_closed(tmp_path) -> None:
    p = tmp_path / "prereg.json"
    p.write_text("{}", encoding="utf-8")
    assert verify_seal(p)["valid"] is False


# -- committed forward pre-registration artifact is sealed --


def test_committed_forward_preregistration_is_sealed() -> None:
    assert COMMITTED.is_file(), "committed forward pre-registration must exist"
    payload = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert payload["status"] == "NOT_YET_COLLECTED"
    assert payload["not_yet_collected"] is True
    assert payload["holdout_statement"]["holdout_2025"] == "SPENT"
    assert payload["window"]["start"] == "2026-07-01"
    assert verify_payload(payload)["valid"] is True
    assert verify_seal(COMMITTED)["valid"] is True
