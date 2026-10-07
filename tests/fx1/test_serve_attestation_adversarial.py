"""SYNTHETIC adversarial probes for ``fx1.serve.attestation``.

Attestation-ladder honesty: the TEE/zkml rungs are structural checks only
(crypto is delegated to platform SDKs at deploy), and they must be
fail-closed — whitespace is no signature, relative artifact claims anchor
at the manifest's own directory, directories/empty paths are not proofs,
and a manifest attesting nothing cannot stand up a rung. All fixtures are
synthetic quotes/manifests, never real attestations.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.serve.attestation import (
    AttestationTier,
    OperatorProofManifest,
    TEEQuote,
    attestation_ladder_status,
    verify_quote,
)

_CKPT = "a" * 64
_OTHER_CKPT = "b" * 64
_NONCE = "synthetic-nonce-1234"


@pytest.fixture(autouse=True)
def _no_signing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic signed_release rung: no key → verify_release raises
    RuntimeError → the ladder catches it fail-closed."""
    monkeypatch.delenv("FX1_SIGNING_KEY", raising=False)


def _quote(**overrides: str) -> TEEQuote:
    fields = {
        "platform": "simulated",
        "checkpoint_sha256": _CKPT,
        "measurement": "deadbeef" * 8,
        "report_data": _NONCE + _CKPT,  # binds: ends with checkpoint sha
        "signature": "synthetic-platform-signature",
    }
    fields.update(overrides)
    return TEEQuote.model_validate(fields)


def _manifest(**overrides: object) -> OperatorProofManifest:
    fields: dict[str, object] = {
        "checkpoint_sha256": _CKPT,
        "covered_operators": ["calibration_head"],
        "proof_artifacts": {"calibration_head": "calibration_head.proof"},
    }
    fields.update(overrides)
    return OperatorProofManifest.model_validate(fields)


def _write_manifest(root: Path, manifest: OperatorProofManifest) -> Path:
    path = root / "zkml.manifest.json"
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    return path


def _write_quote(root: Path, quote: TEEQuote) -> Path:
    path = root / "attestation.quote.json"
    path.write_text(quote.model_dump_json(), encoding="utf-8")
    return path


def test_whitespace_signature_is_no_signature() -> None:
    """'   ' carries no signature material — the structural gate must not
    pass it to the deferred crypto check, and the ladder rung stays down."""
    quote = _quote(signature="   \n\t ")
    assert verify_quote(quote, expected_checkpoint_sha256=_CKPT, nonce=_NONCE) is False


def test_verify_quote_fails_closed_on_every_leg() -> None:
    """Checkpoint mismatch, missing nonce echo, unbound report_data, and a
    short (incidentally-binding) nonce each independently fail the check."""
    good = _quote()
    assert verify_quote(good, expected_checkpoint_sha256=_CKPT, nonce=_NONCE) is True
    # checkpoint mismatch
    foreign = _quote(checkpoint_sha256=_OTHER_CKPT, report_data=_NONCE + _OTHER_CKPT)
    assert verify_quote(foreign, expected_checkpoint_sha256=_CKPT, nonce=_NONCE) is False
    # nonce never echoed into report_data
    no_echo = _quote(report_data="unrelated-material" + _CKPT)
    assert verify_quote(no_echo, expected_checkpoint_sha256=_CKPT, nonce=_NONCE) is False
    # report_data binds checkpoint but never binds the nonce
    bare = _quote(report_data=_CKPT)
    assert verify_quote(bare, expected_checkpoint_sha256=_CKPT, nonce=_NONCE) is False
    # short nonces bind incidentally — refused before the substring check
    short_nonce = _quote(report_data="x" + _CKPT)
    assert verify_quote(short_nonce, expected_checkpoint_sha256=_CKPT, nonce="x") is False
    # empty signature
    assert (
        verify_quote(
            _quote(signature=""),
            expected_checkpoint_sha256=_CKPT,
            nonce=_NONCE,
        )
        is False
    )


def test_quote_binding_accepts_sha256_of_checkpoint_anywhere() -> None:
    """Binding via sha256(checkpoint) embedded mid-report also verifies —
    both binding forms are measured behavior."""
    digested = hashlib.sha256(_CKPT.encode()).hexdigest()
    quote = _quote(report_data=_NONCE + "hdr" + digested)
    assert quote.binds_checkpoint is True
    assert verify_quote(quote, expected_checkpoint_sha256=_CKPT, nonce=_NONCE)


def test_manifest_artifacts_anchor_at_manifest_dir_not_cwd(tmp_path: Path) -> None:
    """A relative proof claim resolves against the manifest's directory via
    the ladder — a same-named file in the process cwd cannot satisfy it."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    manifest = _manifest()
    _write_manifest(ckpt, manifest)
    # Artifact missing relative to manifest; decoy planted in process cwd.
    cwd = Path.cwd()
    decoy = cwd / "calibration_head.proof"
    decoy.write_bytes(b"decoy")
    try:
        status = attestation_ladder_status(ckpt)
        assert status[AttestationTier.SELECTIVE_ZKML.value] is False
        # Plant the real artifact next to the manifest → rung verifies.
        (ckpt / "calibration_head.proof").write_bytes(b"real proof")
        status = attestation_ladder_status(ckpt)
        assert status[AttestationTier.SELECTIVE_ZKML.value] is True
    finally:
        decoy.unlink()


def test_verify_without_root_resolves_against_process_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Called directly with no root, resolution falls back to cwd — measured
    inherited behavior the ladder no longer relies on."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "calibration_head.proof").write_bytes(b"proof")
    manifest = _manifest()
    monkeypatch.chdir(work)
    assert manifest.verify_artifacts_exist() is True
    monkeypatch.chdir(tmp_path)
    assert manifest.verify_artifacts_exist() is False


def test_directory_and_empty_paths_are_not_proof_artifacts(tmp_path: Path) -> None:
    """A directory or '' (Path('.') — the manifest dir itself) previously
    satisfied existence checks; proofs must be real files."""
    (tmp_path / "dirproof").mkdir()
    assert (
        _manifest(
            proof_artifacts={
                "calibration_head": "dirproof",
                "honesty_gate_logits": "",
            },
            covered_operators=["calibration_head", "honesty_gate_logits"],
        ).verify_artifacts_exist(tmp_path)
        is False
    )
    assert (
        _manifest(proof_artifacts={"calibration_head": "dirproof"}).verify_artifacts_exist(tmp_path)
        is False
    )


def test_empty_manifest_cannot_claim_zkml(tmp_path: Path) -> None:
    """A manifest with zero declared artifacts attests nothing — vacuous
    truth must not light the tier-3 rung."""
    manifest = _manifest(covered_operators=[], proof_artifacts={})
    assert manifest.verify_artifacts_exist(tmp_path) is False
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _write_manifest(ckpt, manifest)
    status = attestation_ladder_status(ckpt)
    assert status[AttestationTier.SELECTIVE_ZKML.value] is False


def test_ladder_tee_rung_fails_closed(tmp_path: Path) -> None:
    """Whitespace signatures and corrupt quote files keep the TEE rung down
    — never raise, never verify. The rung checks the quote's SELF-binding
    (report_data echoes the quote's own checkpoint_sha256): which
    checkpoint that is has no directory-level anchor, so cross-checking is
    delegated to the signed release manifest covering the quote file and
    to deploy-time vendor crypto — pinned as measured, not as a defect."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    status = attestation_ladder_status(ckpt)
    assert status == {"signed_release": False, "tee": False, "selective_zkml": False}

    # A quote bound to a checkpoint other than this dir's still self-binds:
    # the rung reports structural validity only (release manifest signs the
    # quote file itself; crypto cross-check lives at deploy).
    foreign = _quote(checkpoint_sha256=_OTHER_CKPT, report_data=_NONCE + _OTHER_CKPT)
    assert foreign.binds_checkpoint is True
    _write_quote(ckpt, foreign)
    assert attestation_ladder_status(ckpt)[AttestationTier.TEE.value] is True

    _write_quote(ckpt, _quote(signature="   "))
    assert attestation_ladder_status(ckpt)[AttestationTier.TEE.value] is False

    (ckpt / "attestation.quote.json").write_text("{not json", encoding="utf-8")
    assert attestation_ladder_status(ckpt)[AttestationTier.TEE.value] is False

    _write_quote(ckpt, _quote())
    assert attestation_ladder_status(ckpt)[AttestationTier.TEE.value] is True


def test_quote_unbound_report_data_never_lights_tee(tmp_path: Path) -> None:
    """A quote whose report_data never echoes its checkpoint sha is not even
    self-consistent — the rung cannot stand it up."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    unbound = _quote(report_data=_NONCE + "no-checkpoint-echo")
    assert unbound.binds_checkpoint is False
    _write_quote(ckpt, unbound)
    assert attestation_ladder_status(ckpt)[AttestationTier.TEE.value] is False


def test_ladder_manifest_rung_fails_closed_on_corrupt_file(tmp_path: Path) -> None:
    """An unparseable zkml manifest claims nothing — rung False, no raise."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "zkml.manifest.json").write_text("{not json", encoding="utf-8")
    status = attestation_ladder_status(ckpt)
    assert status[AttestationTier.SELECTIVE_ZKML.value] is False


def test_schemas_reject_unknown_platforms_and_smuggled_fields() -> None:
    """Closed schemas: platform allowlist, exact-length checkpoint, and
    coverage claims must have matching artifact entries."""
    with pytest.raises(ValidationError):
        _quote(platform="aws-nitro")  # not on the platform allowlist
    with pytest.raises(ValidationError):
        _quote(checkpoint_sha256="short")
    with pytest.raises(ValidationError):
        # covered operator with no proof artifact — declared coverage gap
        _manifest(covered_operators=["calibration_head", "unproven_op"])
    manifest = _manifest(proof_artifacts={"calibration_head": "p", "extra_op": "e"})
    assert "extra_op" in manifest.proof_artifacts  # spare proofs allowed
