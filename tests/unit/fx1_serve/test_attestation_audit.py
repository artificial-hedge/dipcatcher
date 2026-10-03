"""attestation_audit lane: fx-1 attestation contract pinned."""

from __future__ import annotations

from fx1.serve.attestation_audit import attestation_audit, attestation_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_attestation_contract() -> None:
    r = attestation_audit()
    q, rel = r["quote"], r["release"]
    assert q["honest_quote_verifies"] is True
    assert q["wrong_checkpoint"] is False
    assert q["nonce_missing"] is False
    assert q["empty_signature"] is False
    assert q["one_char_nonce_binds"] is False  # fixed: <8-char nonces fail closed
    assert rel["honest_release_verifies"] is True
    assert rel["tampered_artifact_fails"] is True
    assert rel["forged_signature_fails"] is True
    assert rel["missing_manifest_fails"] is True
    assert rel["unlisted_extra_file_passes"] is True  # flagged coverage gap
    assert r["key_gate"]["unset_fails_closed"] == "raise:RuntimeError"
    assert r["zkml_manifest"]["claimed_coverage_without_proof"] == "raise:ValueError"


def test_bench_ok_and_verifies() -> None:
    receipt = attestation_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "attestation_audit_test.json")
    assert result["valid"], result.get("errors")
