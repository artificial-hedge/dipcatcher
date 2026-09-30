"""``release_attestation.v1`` — bind shipped artifact bytes to verified state.

The integrity stack proves the *repo*; this lane extends the proof to bytes
that live outside it (a wheel, an sdist tarball, an exported evidence bundle).
An attestation records each artifact's sha256 + size, digests of the pin
state at attestation time (checkpoint, pins, repo_integrity attestation),
the emitting git revision, and an Ed25519 signature over the whole body —
the same key that signs the pin manifest, so rotation lineage applies.

An auditor with the wheel + the attestation + the committed pubkey can prove
"these bytes shipped from a tree whose gates were green", without trusting a
build host: ``verify_release_attestation`` re-hashes the artifacts, verifies
the signature, and — when a repo root is supplied — checks the recorded pin
digests against the live tree (``attestation_stale`` when they've moved on,
which invalidates *freshness* but not the signature's authenticity).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

RELEASE_ATTESTATION_SCHEMA = "release_attestation.v1"

# Pin state digested into every attestation — freshness anchors.
_PINNED_STATE = (
    "quality/checkpoint.json",
    "quality/epoch_heads.json",
    "quality/crown_jewels.json",
    "quality/repo_integrity.json",
)


def release_attestation(
    artifacts: Mapping[str, bytes],
    *,
    root: Path | str = ".",
    private_seed_hex: str,
    pubkey_hex: str,
) -> dict[str, Any]:
    """Build a signed ``release_attestation.v1`` for ``artifacts``.

    ``artifacts`` maps artifact name → raw bytes (wheels, sdists, bundles —
    whatever shipped). Requires the pin files to exist: an attestation over
    an un-pinned tree attests nothing.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    root_p = Path(root)
    state: dict[str, str] = {}
    for rel in _PINNED_STATE:
        p = root_p / rel
        if p.is_file():
            state[rel] = hash_bytes(p.read_bytes())
    if "quality/checkpoint.json" not in state or "quality/epoch_heads.json" not in state:
        raise ValueError("release attestation requires a checkpointed, pinned tree")

    body: dict[str, Any] = {
        "kind": RELEASE_ATTESTATION_SCHEMA,
        "schema": RELEASE_ATTESTATION_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "artifacts": {
            name: {"sha256": hash_bytes(data), "n_bytes": len(data)}
            for name, data in sorted(artifacts.items())
        },
        "pinned_state_sha256": state,
        "generated_at_commit": _git_rev(root_p),
        "algorithm": "ed25519",
        "key_id": _key_id(pubkey_hex),
    }
    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_seed_hex))
    body["signature"] = key.sign(canonical_json_bytes(body)).hex()
    return body


def _key_id(pubkey_hex: str) -> str:
    from quant_fund.research.gate_signatures import key_id

    return key_id(pubkey_hex)


def _git_rev(root: Path) -> str | None:
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 — garnish, never a gate
        return None


def release_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``release_attestation.v1`` structural coherence; ``[]`` when clean."""
    errors: list[str] = []
    if payload.get("kind") != RELEASE_ATTESTATION_SCHEMA:
        errors.append("kind")
    if payload.get("schema") != RELEASE_ATTESTATION_SCHEMA:
        errors.append("schema")
    arts = payload.get("artifacts")
    if not isinstance(arts, Mapping) or not arts:
        errors.append("artifacts_empty")
    else:
        for name, meta in arts.items():
            if (
                not isinstance(meta, Mapping)
                or not isinstance(meta.get("sha256"), str)
                or len(meta.get("sha256") or "") != 64
                or not isinstance(meta.get("n_bytes"), int)
            ):
                errors.append(f"artifact_malformed:{name}")
    state = payload.get("pinned_state_sha256")
    if not isinstance(state, Mapping) or "quality/checkpoint.json" not in state:
        errors.append("pinned_state_sha256")
    if payload.get("algorithm") != "ed25519":
        errors.append("algorithm")
    if not isinstance(payload.get("key_id"), str):
        errors.append("key_id")
    sig = payload.get("signature")
    if not (isinstance(sig, str) and len(sig) == 128):
        errors.append("signature_malformed")
    return errors


def verify_release_attestation(
    payload: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
    *,
    root: Path | str | None = None,
    pubkey_hex: str | None = None,
) -> dict[str, Any]:
    """Verify a release attestation against artifact bytes.

    Always checks: contract, per-artifact sha256+size, Ed25519 signature
    under ``pubkey_hex`` (or the committed ``quality/gate_signing.pub`` when
    ``root`` is given — so rotation-era attestations verify via the keyring).
    With ``root``: the recorded pin digests are compared to the live tree —
    ``attestation_stale`` when they've drifted.
    """
    errors = release_contract_errors(payload)
    if errors:
        return {"ok": False, "errors": [f"contract:{e}" for e in errors]}
    errs: list[str] = []

    declared = payload["artifacts"]
    assert isinstance(declared, Mapping)
    if set(artifacts) != set(declared):
        errs.append("artifact_set_mismatch")
    for name, data in artifacts.items():
        meta = declared.get(name)
        if not isinstance(meta, Mapping):
            continue
        if hash_bytes(data) != meta.get("sha256") or len(data) != meta.get("n_bytes"):
            errs.append(f"artifact_digest_mismatch:{name}")

    pub = pubkey_hex
    if pub is None and root is not None:
        pub_file = Path(root) / "quality/gate_signing.pub"
        if pub_file.is_file():
            pub = pub_file.read_text().strip()
    if pub is None:
        errs.append("pubkey_unavailable")
    else:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        try:
            key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub.strip()))
        except ValueError:
            errs.append("pubkey_malformed")
            key = None
        if key is not None:
            if payload.get("key_id") != _key_id(pub.strip()):
                errs.append("key_id_mismatch")
            unsigned = {k: v for k, v in payload.items() if k != "signature"}
            try:
                key.verify(
                    bytes.fromhex(str(payload["signature"])),
                    canonical_json_bytes(unsigned),
                )
            except InvalidSignature:
                errs.append("signature_invalid")

    if root is not None:
        state = payload.get("pinned_state_sha256")
        assert isinstance(state, Mapping)
        stale: list[str] = []
        for rel, digest in state.items():
            p = Path(root) / str(rel)
            if not p.is_file() or hash_bytes(p.read_bytes()) != digest:
                stale.append(str(rel))
        if stale:
            errs.append(f"attestation_stale:{','.join(sorted(stale))}")
    return {"ok": not errs, "errors": errs}
