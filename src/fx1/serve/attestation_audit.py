"""attestation_audit — adversarial probes on the fx-1 attestation ladder.

Two previously demonstrated edges now fail closed:

1. ``verify_quote`` anti-replay was a bare *substring* check — ``nonce in
   report_data``. A 1-character nonce bound to any quote whose report_data
   happened to contain it; a nonce that was never echoed passed. Fixed:
   ``_MIN_NONCE_LEN = 8`` — short nonces fail closed.
2. ``verify_release`` checks the complete regular-file inventory. Adding
   an unlisted checkpoint artifact invalidates the release; the audit
   fails if manifest-scoped verification ever returns.

Also pinned: ``_key()`` refuses an unset env (fail closed), a forged
signature fails, a tampered artifact fails, ``OperatorProofManifest``
rejects claimed coverage without proof paths. Sealed
``attestation_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["attestation_audit", "attestation_audit_bench"]

_KEY_ENV = "FX1_SIGNING_KEY"
# keep in sync with signing.SIGNING_KEY_ENV


def _probe_quote() -> dict[str, Any]:
    from fx1.serve.attestation import TEEQuote, verify_quote

    ckpt = hashlib.sha256(b"weights").hexdigest()
    good = TEEQuote(
        platform="sev-snp",
        checkpoint_sha256=ckpt,
        measurement="aa" * 48,
        report_data=f"nonce-xyz::{ckpt}",
        signature="sig",
    )
    out = {
        "honest_quote_verifies": verify_quote(
            good, expected_checkpoint_sha256=ckpt, nonce="nonce-xyz"
        ),
        "wrong_checkpoint": verify_quote(
            good, expected_checkpoint_sha256="f" * 64, nonce="nonce-xyz"
        ),
        "nonce_missing": verify_quote(good, expected_checkpoint_sha256=ckpt, nonce="never-echoed"),
        "empty_signature": verify_quote(
            good.model_copy(update={"signature": ""}),
            expected_checkpoint_sha256=ckpt,
            nonce="nonce-xyz",
        ),
    }
    # Substring weakness: 1-char nonce binds wherever it appears.
    weak = TEEQuote(
        platform="sev-snp",
        checkpoint_sha256=ckpt,
        measurement="aa" * 48,
        report_data=f"report::{ckpt}",
        signature="sig",
    )
    out["one_char_nonce_binds"] = verify_quote(weak, expected_checkpoint_sha256=ckpt, nonce="r")
    return out


def _probe_release() -> dict[str, Any]:
    from fx1.serve.signing import sign_release, verify_release

    saved = os.environ.get(_KEY_ENV)
    os.environ[_KEY_ENV] = "audit-scratch-key"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "weights.bin").write_bytes(b"fake-weights")
            (root / "config.json").write_text("{}")
            sign_release(root)
            honest = verify_release(root)

            (root / "weights.bin").write_bytes(b"tampered")
            tampered = verify_release(root)

            # fresh dir for the unsigned-addition regression
            root2 = Path(tmp) / "ckpt2"
            root2.mkdir()
            (root2 / "weights.bin").write_bytes(b"w")
            sign_release(root2)
            (root2 / "smuggled.bin").write_bytes(b"unlisted")
            unlisted_extra = verify_release(root2)

            # forged signature
            root3 = Path(tmp) / "ckpt3"
            root3.mkdir()
            (root3 / "weights.bin").write_bytes(b"w")
            sign_release(root3)
            (root3 / "release.sig").write_text("f" * 64)
            forged = verify_release(root3)

            # missing manifest
            root4 = Path(tmp) / "ckpt4"
            root4.mkdir()
            (root4 / "weights.bin").write_bytes(b"w")
            missing = verify_release(root4)
        return {
            "honest_release_verifies": honest,
            "tampered_artifact_fails": not tampered,
            "unlisted_extra_file_passes": unlisted_extra,
            "forged_signature_fails": not forged,
            "missing_manifest_fails": not missing,
        }
    finally:
        if saved is None:
            os.environ.pop(_KEY_ENV, None)
        else:
            os.environ[_KEY_ENV] = saved


def attestation_audit() -> dict[str, Any]:
    results: dict[str, Any] = {}
    results["quote"] = _probe_quote()
    results["release"] = _probe_release()

    from fx1.serve.signing import _key  # noqa: SLF001 — auditing the private gate

    saved = os.environ.pop(_KEY_ENV, None)
    try:
        try:
            _key()
            unset = "ok"
        except RuntimeError:
            unset = "raise:RuntimeError"
    finally:
        if saved is not None:
            os.environ[_KEY_ENV] = saved
    results["key_gate"] = {"unset_fails_closed": unset}

    from fx1.serve.attestation import OperatorProofManifest

    try:
        OperatorProofManifest(
            checkpoint_sha256="a" * 64,
            covered_operators=["honesty_gate"],
            proof_artifacts={},
        )
        coverage = "accepted"
    except ValueError:
        coverage = "raise:ValueError"
    results["zkml_manifest"] = {"claimed_coverage_without_proof": coverage}
    return results


def attestation_audit_bench() -> dict[str, Any]:
    r = attestation_audit()
    q, rel, key, zk = r["quote"], r["release"], r["key_gate"], r["zkml_manifest"]
    ok = (
        q["honest_quote_verifies"] is True
        and q["wrong_checkpoint"] is False
        and q["nonce_missing"] is False
        and q["empty_signature"] is False
        and rel["honest_release_verifies"] is True
        and rel["tampered_artifact_fails"] is True
        and rel["forged_signature_fails"] is True
        and rel["missing_manifest_fails"] is True
        and key["unset_fails_closed"] == "raise:RuntimeError"
        and zk["claimed_coverage_without_proof"] == "raise:ValueError"
    )
    flags = {
        "quote_substring_nonce": q["one_char_nonce_binds"],
        "unlisted_extra_verifies": rel["unlisted_extra_file_passes"],
    }
    ok = (
        ok and flags["quote_substring_nonce"] is False and flags["unlisted_extra_verifies"] is False
    )
    payload: dict[str, Any] = {
        "kind": "attestation_audit",
        "schema": "attestation_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "flags": flags, "ok": ok},
        "interpretation": (
            "Attestation contract holds on structural edges; substring "
            "nonce binding fails closed below 8 chars, and releases "
            "reject unlisted checkpoint artifacts. These SYNTHETIC "
            "checks do not verify vendor signatures or prove TEE execution."
            if ok
            else f"ATTESTATION DEFECT: {r}"
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
