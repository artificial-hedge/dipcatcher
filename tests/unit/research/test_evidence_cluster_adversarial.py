"""Adversarial probes for the evidence/receipt-graph audit cluster.

SYNTHETIC only — every fixture is constructed in tmp dirs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from quant_fund.research import (
    corpus_inference,
    impossible_fit,
    receipt_graph,
    release_attestation,
    script_receipts,
    tape_registry,
)
from quant_fund.research.coherence import coherence_contract_errors
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def _manifest(tmp: Path, tape_entry: dict[str, Any]) -> Path:
    body: dict[str, Any] = {
        "schema": tape_registry.TAPE_MANIFEST_SCHEMA,
        "source_label": "probe",
        "tape_files": [tape_entry],
        "frame_csv_sha256": "0" * 64,
        "n_rows": 1,
        "n_names": 1,
        "window": None,
    }
    manifest = {**body, "receipt_sha256": hash_bytes(canonical_json_bytes(body))}
    path = tmp / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_tape_manifest_symlinked_tape_is_uncontained(tmp_path: Path) -> None:
    """A manifest-declared tape path that resolves outside root via a symlink
    must be refused — not hashed into a sha256 oracle."""
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"not a tape")
    root = tmp_path / "repo"
    root.mkdir()
    link = root / "tape.parquet"
    link.symlink_to(outside)
    manifest = _manifest(
        root,
        {
            "path": "tape.parquet",
            "sha256": hash_bytes(outside.read_bytes()),
            "n_bytes": outside.stat().st_size,
        },
    )
    errors = tape_registry.verify_manifest(manifest, root)
    assert any(e == "tape_files[0]_path_uncontained" for e in errors), errors


def _attestation() -> dict[str, Any]:
    return {
        "kind": release_attestation.RELEASE_ATTESTATION_SCHEMA,
        "schema": release_attestation.RELEASE_ATTESTATION_SCHEMA,
        "artifacts": {"wheel.whl": {"sha256": "0" * 64, "n_bytes": 4}},
        "pinned_state_sha256": {"quality/checkpoint.json": "0" * 64},
        "algorithm": "ed25519",
        "key_id": "k",
        "signature": "z" * 128,  # contract-clean length, non-hex
    }


def test_release_attestation_nonhex_signature_is_invalid_not_crash() -> None:
    payload = _attestation()
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    pub = Ed25519PrivateKey.generate().public_key().public_bytes_raw().hex()
    result = release_attestation.verify_release_attestation(
        payload, {"wheel.whl": b"data"}, pubkey_hex=pub
    )
    assert not result["ok"]
    assert "signature_invalid" in result["errors"]


def test_release_attestation_pinned_state_cannot_traverse(tmp_path: Path) -> None:
    payload = _attestation()
    payload["pinned_state_sha256"]["../../etc/hostname"] = "0" * 64
    result = release_attestation.verify_release_attestation(
        payload, {"wheel.whl": b"data"}, root=tmp_path, pubkey_hex="aa" * 32
    )
    assert any(
        e.startswith("pinned_state_path_invalid:../../etc/hostname") for e in result["errors"]
    )


def test_script_receipt_contract_crash_reads_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(payload: object) -> list[str]:
        raise IndexError("hostile input")

    monkeypatch.setitem(script_receipts.SCRIPT_RECEIPT_CONTRACTS, "probe.v1", _boom)
    assert script_receipts.script_receipt_contract_errors("probe.v1", {}) == [
        "contract_check_error"
    ]


def _receipt(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")


def test_corpus_audit_nested_members_and_traversal(tmp_path: Path) -> None:
    """Nested non-quarantined receipts are corpus members, and a pinned member
    name that escapes the corpus root is a recorded error, not a silent read."""
    receipts = tmp_path / "receipts"
    _receipt(
        receipts / "sub" / "nested.json",
        {"kind": "probe.v1", "data_label": "SYNTHETIC", "p_value": 0.01},
    )
    _receipt(
        receipts / "top.json",
        {"kind": "probe.v1", "data_label": "SYNTHETIC", "p_value": 0.02},
    )
    result = corpus_inference.corpus_audit(receipts)
    assert "sub/nested.json" in result["params"]["input_labels"]
    pinned = corpus_inference.corpus_audit(receipts, members={"../outside.json"})
    assert any(e["error"] == "member_path_uncontained" for e in pinned["parse_errors"]), pinned[
        "parse_errors"
    ]


def test_receipt_graph_nested_member_resolution(tmp_path: Path) -> None:
    """A receipt in a non-quarantined subdir is a graph member; a basename ref
    resolves it when unique and stays unresolvable when ambiguous."""
    root = tmp_path / "receipts"
    _receipt(
        root / "sub" / "deep.json",
        {"kind": "x.v1", "data_label": "CORPUS"},
    )
    _receipt(
        root / "citer.json",
        {"kind": "x.v1", "data_label": "CORPUS", "prev_receipt": "deep.json"},
    )
    result = receipt_graph.receipt_graph(root)
    assert "sub/deep.json" in result["members"]
    edge = next(e for e in result["edges"] if e["file"] == "citer.json")
    assert edge["target"] == "sub/deep.json"
    assert receipt_graph.graph_contract_errors(result) == []


def test_receipt_graph_ambiguous_basename_stays_unresolvable(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    for sub in ("a", "b"):
        _receipt(root / sub / "dup.json", {"kind": "x.v1", "data_label": "CORPUS"})
    _receipt(
        root / "citer.json",
        {"kind": "x.v1", "data_label": "CORPUS", "prev_receipt": "dup.json"},
    )
    result = receipt_graph.receipt_graph(root)
    assert result["n_unresolvable"] == 1
    assert result["verdict"] == "dangling"


def test_impossible_fit_second_matching_key_not_masked() -> None:
    """A benign first pinball key must not hide an impossible sibling zero."""
    metrics = {"pinball_a": 0.5, "pinball_b": 0.0, "n": 200}
    flags = impossible_fit.impossible_fit_flags(metrics)
    assert "exact_zero:pinball_b" in flags


def test_coherence_contract_rederives_error_count() -> None:
    receipt = {
        "schema": "receipt.v2",
        "payload": {
            "schema": "coherence_eval.v1",
            "kind": "coherence_eval",
            "data_label": "SYNTHETIC",
            "live_pnl_claim": False,
            "results": [{"status": "error"}, {"status": "ok"}],
            "n_error_rows": 0,
        },
    }
    assert "n_error_rows_mismatch" in coherence_contract_errors(receipt)
