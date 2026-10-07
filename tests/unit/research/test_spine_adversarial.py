"""Adversarial probes for the research integrity spine.

Each probe pins a concrete attack the line-level audit found and fixed —
contract-dispatch evasion, verifier crash on hostile input, unverified
lineage injection, and traversal during untrusted-bundle materialization.
Probes assert the fail-closed verdict; none weaken product behavior.
"""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path
from typing import Any

from quant_fund.research import receipt_v2
from quant_fund.research.custody import custody_contract_errors, verify_custody_bundle
from quant_fund.research.integrity_witness import (
    WITNESS_SCHEMA,
    _merkle_leaf,
    verify_witness_record,
)
from quant_fund.research.key_rotation import load_keyring
from quant_fund.research.lane_contracts import lane_contract_errors
from quant_fund.research.quorum_rotation import load_registry_lineage
from quant_fund.research.receipt_v2 import seal_receipt, verify_receipt_payload
from quant_fund.utils.hashing import hash_bytes

_B64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
_HEX64 = "ab" * 32


def _v2_envelope(inner: dict[str, Any], *, kind: str) -> dict[str, Any]:
    """A sealed receipt.v2 envelope whose envelope kind claims ``kind``
    while the sealed inner payload claims its own identity."""
    return seal_receipt(
        receipt_v2.build_receipt_v2(
            kind=kind,
            data_label="SYNTHETIC",
            dataset={"probe": "x"},
            params={"probe": 1},
            code_files=(Path(receipt_v2.__file__),),
            verdict="pass",
            payload=inner,
        )
    )


def _witness_record(*, signature: object = None, note: str) -> dict[str, Any]:
    """Minimal integrity_witness.v1 record reaching the signature + note
    checks: logged digest binds the supplied target bytes, the UUID tail is
    the real merkle leaf, and a one-leaf inclusion proof recomputes. The
    checkpoint note lives on the inclusion proof, as Rekor commits it."""
    digest = hash_bytes(b"target-bytes")
    spec: dict[str, Any] = {"data": {"hash": {"value": digest}}}
    if signature is not None:
        spec["signature"] = signature
    body_b64 = _B64(json.dumps({"spec": spec}).encode())
    leaf = _merkle_leaf(body_b64)
    return {
        "schema": WITNESS_SCHEMA,
        "target": {"file": "quality/checkpoint.json", "sha256": digest},
        "rekor": {
            "body_b64": body_b64,
            "uuid": "00" * 16 + leaf.hex(),
            "inclusion_proof": {
                "log_index": 0,
                "tree_size": 1,
                "hashes": [],
                "root_hash": leaf.hex(),
                "checkpoint_note": note,
            },
            "signed_entry_timestamp": "junk",
            "integrated_time": 0,
            "log_id": "x",
            "log_index": 0,
        },
    }


# --- contract-dispatch evasion --------------------------------------------


def test_v2_inner_kind_claim_routes_deep_lane_check() -> None:
    """Inner payload claims ``hstep_bench`` inside the seal; the envelope
    kind is renamed to a benign string. The hstep consistency lane must
    still fire — before the fix the renamed envelope skipped it entirely."""
    inner = {"kind": "hstep_bench", "verdict": "pass", "h_values": [1], "horizons": {}}
    receipt = _v2_envelope(inner, kind="innocuous")
    res = verify_receipt_payload(receipt)
    assert "kind_fingerprint_mismatch" in res["errors"]


def test_v2_inner_schema_claim_routes_deep_lane_check() -> None:
    """A ``<base>.v1`` schema claim inside the seal maps to the lane kind —
    schema claims cannot hide the bespoke lane checks either."""
    inner = {"schema": "calibration_eval.v1", "kind": "not_calibration"}
    receipt = _v2_envelope(inner, kind="innocuous")
    res = verify_receipt_payload(receipt)
    assert "kind_fingerprint_mismatch" in res["errors"]


def test_v1_evalue_kind_does_not_swallow_capacity_schema() -> None:
    """kind in the evalue family + schema=capacity_overlay.v1: both claims
    are checked — the elif chain used to let the kind branch swallow the
    capacity contract entirely."""
    payload = seal_receipt(
        {
            "kind": "tail_audit",
            "schema": "capacity_overlay.v1",
            "live_pnl_claim": False,
            "results": [],
        }
    )
    res = verify_receipt_payload(payload)
    assert "results_missing" in res["errors"]


def test_lane_contract_dual_claim_runs_every_contract() -> None:
    """schema=capacity_overlay.v1 + kind=sim_live_receipt: the honesty
    contract must run even though an earlier schema dispatch matched —
    first-match dispatch made the sim_live flags unreachable."""
    payload = {
        "schema": "capacity_overlay.v1",
        "kind": "sim_live_receipt",
        "results": [],
        "research_only": True,
        "live_pnl_claim": False,
    }
    errors = lane_contract_errors(payload)
    assert "results_missing" in errors  # capacity contract ran
    assert "simulated_only" in errors  # sim_live honesty contract ran too


def test_p42_code_sha256_confined_to_repo() -> None:
    """``code_sha256`` paths resolve under the repo root — a ``../../`` path
    must not probe (let alone pass on) out-of-repo file existence."""
    payload = {
        "receipt": "fast_replay_p42_conformance",
        "research_only": True,
        "live_pnl_claim": False,
        "verdict": "pass",
        "disclaimer": "SYNTHETIC evidence",
        "base_commit": "0" * 40,
        "evidence": ["x"],
        "fixes": [],
        "gaps_refused": [],
        # /etc/passwd exists on disk — pre-fix this probed it and passed.
        "code_sha256": {"../../../../../../etc/passwd": _HEX64},
    }
    errors = lane_contract_errors(payload)
    assert "code_sha256:../../../../../../etc/passwd_missing_file" in errors


# --- hostile-input crash surfaces ------------------------------------------


def test_witness_record_malformed_note_fails_not_crashes() -> None:
    """A checkpoint_note whose signature block is not base64 raised
    ``binascii.Error`` out of the verifier — a hostile proof could DoS the
    gate. It must report signature-invalid instead."""
    record = _witness_record(signature={"content": "AA=="}, note="note\n\n !!!!")
    res = verify_witness_record(
        record,
        target_bytes=b"target-bytes",
        witness_pubkey_pem=b"not-a-key",
        rekor_pubkey_pem=b"not-a-key",
    )
    assert "checkpoint_note_signature_invalid" in res["errors"]


def test_witness_record_missing_signature_fails_closed() -> None:
    """A spec carrying no ``signature`` block proved no attribution — the
    isinstance guard used to skip the check silently."""
    record = _witness_record(signature=None, note="note\n\njunk")
    res = verify_witness_record(
        record,
        target_bytes=b"target-bytes",
        witness_pubkey_pem=b"not-a-key",
        rekor_pubkey_pem=None,
    )
    assert "witness_signature_invalid" in res["errors"]


# --- unverified lineage injection ------------------------------------------


def test_keyring_rejects_unauthorized_rotation_record(tmp_path: Path) -> None:
    """A well-formed but unsigned ``rotation_*.json`` must not inject its
    pubkeys into the keyring checkpoint verification trusts."""
    injected_pub = "cd" * 32
    forged = {
        "schema": "key_rotation.v1",
        "payload": {
            "old_pubkey": "ef" * 32,
            "new_pubkey": injected_pub,
            "old_key_id": "forged",
            "new_key_id": "forged",
        },
        "old_signature": "00",
        "new_signature": "00",
    }
    (tmp_path / "quality").mkdir()
    (tmp_path / "quality" / "rotation_forge.json").write_text(json.dumps(forged))
    from quant_fund.research.gate_signatures import key_id

    ring = load_keyring(tmp_path)
    assert key_id(injected_pub) not in ring


def test_quorum_lineage_rejects_unauthorized_rotation(tmp_path: Path) -> None:
    """Same class for quorum registries: a forged ``quorum_rotations``
    record must not admit an attacker registry into v2 signature
    resolution."""
    from quant_fund.research.gate_signatures import key_id, registry_sha256

    attacker_pub = "cd" * 32
    attacker_reg = {
        "schema": "gate_quorum.v1",
        "keys": [{"key_id": key_id(attacker_pub), "pubkey": attacker_pub}],
        "threshold": 1,
    }
    forged = {
        "schema": "quorum_rotation.v1",
        "payload": {
            "prev_registry": attacker_reg,
            "prev_registry_sha256": registry_sha256(attacker_reg),
            "registry": attacker_reg,
            "registry_sha256": registry_sha256(attacker_reg),
        },
        "signatures": [{"key_id": key_id(attacker_pub), "signature": "00"}],
    }
    rot_dir = tmp_path / "quality" / "quorum_rotations"
    rot_dir.mkdir(parents=True)
    (rot_dir / "rotation_forge.json").write_text(json.dumps(forged))

    lineage = load_registry_lineage(tmp_path)
    assert registry_sha256(attacker_reg) not in lineage


# --- untrusted-bundle materialization --------------------------------------


def _custody_bundle(**overrides: Any) -> dict[str, Any]:
    """A contract-valid custody bundle (a single-hop chain where the one
    epoch receipt is both first and head)."""
    hop_raw = json.dumps({"members": []}).encode()
    bundle: dict[str, Any] = {
        "kind": "custody_proof.v1",
        "schema": "custody_proof.v1",
        "member": "member.json",
        "member_sha256": _HEX64,
        "corpus_dir": "receipts",
        "pattern": "*.json",
        "first_epoch": "corpus_epoch_0.json",
        "chain_head": "corpus_epoch_0.json",
        "hops": [{"name": "corpus_epoch_0.json", "sha256": "12" * 32, "bytes_b64": _B64(hop_raw)}],
        "inclusion": {},
        "embedded_files": {},
        "witness_proofs": {},
    }
    bundle.update(overrides)
    return bundle


def test_custody_contract_flags_embedded_traversal() -> None:
    bundle = _custody_bundle(
        embedded_files={"../../custody_escape_marker": _B64(b"owned")},
    )
    errors = custody_contract_errors(bundle)
    assert any("embedded_path_uncontained" in e for e in errors)


def test_custody_verify_never_writes_outside_sandbox() -> None:
    """corpus_dir traversal in a contract-valid bundle: verify must report
    the containment failure and create nothing outside its temp root."""
    marker = Path(tempfile.gettempdir()) / "custody_escape_dir"
    assert not marker.exists()
    bundle = _custody_bundle(corpus_dir="../custody_escape_dir")
    res = verify_custody_bundle(bundle, member_bytes=b"member bytes")
    assert res["ok"] is False
    assert "corpus_dir_uncontained" in res["errors"]
    assert not marker.exists()


def test_custody_verify_rejects_traversal_hop_name() -> None:
    marker = Path(tempfile.gettempdir()) / "custody_hop_escape.json"
    assert not marker.exists()
    hop_raw = json.dumps({"members": []}).encode()
    bundle = _custody_bundle(
        first_epoch="../../custody_hop_escape.json",
        chain_head="../../custody_hop_escape.json",
        hops=[
            {
                "name": "../../custody_hop_escape.json",
                "sha256": "12" * 32,
                "bytes_b64": _B64(hop_raw),
            }
        ],
    )
    res = verify_custody_bundle(bundle, member_bytes=b"member bytes")
    assert res["ok"] is False
    assert any("hop_path_uncontained" in e for e in res["errors"])
    assert not marker.exists()


def test_auditor_bundle_rejects_prefixed_traversal_member() -> None:
    """A member key that passes the ``SPINE_PREFIXES`` startswith check
    while still escaping via ``..`` must never be written to disk."""
    from quant_fund.research.auditor_bundle import BUNDLE_MEMBERS, verify_bundle

    marker = Path(tempfile.gettempdir()) / "auditor_escape_marker.json"
    assert not marker.exists()
    files = {rel: _B64(b"junk") for rel in BUNDLE_MEMBERS}
    files["quality/checkpoints/../../../auditor_escape_marker.json"] = _B64(b"owned")
    bundle = {
        "schema": "auditor_bundle.v1",
        "files": files,
        "witness_proof": {"rekor": {"body_b64": _B64(json.dumps({}).encode())}},
        "files_sha256": {},
    }
    bundle_path = Path(tempfile.gettempdir()) / "auditor_probe_bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    try:
        res = verify_bundle(bundle_path, rekor_url=None)
    finally:
        bundle_path.unlink(missing_ok=True)
    assert res["ok"] is False
    assert any("member_path_uncontained" in e for e in res["errors"])
    assert not marker.exists()
