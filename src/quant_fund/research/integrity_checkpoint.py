"""Integrity checkpoint: a portable signed snapshot of the pin state.

``verify-repo`` re-derives every gate from the live tree — powerful, but the
verifier needs the full repo *and* the full stack. A checkpoint instead binds
the whole integrity state into one small artifact: the sha256 of every pin
file (``epoch_heads.json``, ``crown_jewels.json``, ``gate_pins.sig``) plus
each corpus's pinned head receipt, signed with the same Ed25519 key as the
pins themselves and timestamp-anchored like them.

An external auditor's minimal bundle is therefore ``quality/checkpoint.json``
+ ``quality/gate_signing.pub`` + the checkpoint's TSA token: verify the
signature offline (no repo), then walk into a clone and confirm the pinned
files hash to the declared digests. This is the signed-tree-head pattern from
public transparency logs applied to the repo's own pin state.

``verify_checkpoint`` splits *authenticity* (``ok`` — signature valid,
well-formed, TSA-anchored) from *currency* (``current`` — the pinned files
still match the live tree). A stale-but-authentic checkpoint is still valid
proof of the state at its timestamp.

Provenance evidence only; never a market or P&L claim.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_fund.research.corpus_epoch import load_heads_pin
from quant_fund.research.gate_signatures import DEFAULT_PUBKEY_PATH, key_id
from quant_fund.research.timestamp_anchor import stamp_timestamp, verify_timestamps
from quant_fund.utils.atomicio import atomic_write_text
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

CHECKPOINT_SCHEMA = "integrity_checkpoint.v1"
CHECKPOINT_SIG_SCHEMA = "integrity_checkpoint_sig.v1"
DEFAULT_CHECKPOINT_PATH = Path("quality/checkpoint.json")
# Superseded checkpoints are archived append-only under this dir so the
# prev_sha256 chain is locally verifiable — not just digest-referenced via
# the Rekor proofs. The epoch corpus exempts this prefix (like witness/):
# archiving must not force a re-stamp on every checkpoint rotation.
CHECKPOINT_ARCHIVE_DIR = Path("quality/checkpoints")

# The pin files whose bytes the checkpoint covers. gate_pins.sig is included
# so the checkpoint also binds *which* pin signature was current.
PINNED_FILES = (
    "quality/epoch_heads.json",
    "quality/crown_jewels.json",
    "gate_pins.sig",
)


def checkpoint_state(root: str | Path) -> dict[str, Any]:
    """Digest of the pin layer + the chain heads it records."""
    root_path = Path(root)
    pins: dict[str, str] = {}
    missing: list[str] = []
    for rel in PINNED_FILES:
        member = root_path / rel
        if member.exists():
            pins[rel] = hash_bytes(member.read_bytes())
        else:
            missing.append(rel)
    heads_pin = root_path / "quality/epoch_heads.json"
    heads = load_heads_pin(heads_pin) if heads_pin.is_file() else {}
    # Temporal binding: the new checkpoint names the digest of the one it
    # replaces. Each witnessed digest already sits in the public Rekor log,
    # so the checkpoint sequence is anchored end-to-end — a rewritten
    # "current" checkpoint can't claim continuity it never had.
    prev_file = root_path / DEFAULT_CHECKPOINT_PATH
    prev_sha256 = hash_bytes(prev_file.read_bytes()) if prev_file.exists() else None
    from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256

    return {
        "schema": CHECKPOINT_SCHEMA,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pins": pins,
        "heads": heads,
        "missing_pins": missing,
        "prev_sha256": prev_sha256,
        # The pins are only as strong as the verifier that minted them —
        # record WHICH code produced this state: HEAD revision plus a
        # fingerprint over tracked diffs + untracked files. A tampered
        # verifier that re-pins corrupted artifacts can't reproduce the
        # code fingerprint its checkpoint committed to.
        "code": {
            "revision": git_revision(),
            "worktree_sha256": git_worktree_sha256(),
        },
    }


def write_checkpoint(
    root: str | Path,
    private_seed_hex: str,
    pubkey_hex: str,
    *,
    path: Path = DEFAULT_CHECKPOINT_PATH,
) -> Path:
    """Sign ``checkpoint_state`` and write the checkpoint file atomically."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    root_path = Path(root)
    prev_file = root_path / path
    # Archive the superseded checkpoint before overwrite — append-only, keyed
    # by its digest so the prev_sha256 chain can be walked offline.
    if prev_file.exists():
        prev_bytes = prev_file.read_bytes()
        archive_dir = root_path / CHECKPOINT_ARCHIVE_DIR
        archive_dir.mkdir(parents=True, exist_ok=True)
        digest = hash_bytes(prev_bytes)
        archived = archive_dir / f"{digest[:16]}_{digest[16:24]}.json"
        if not archived.exists():
            atomic_write_text(archived, prev_bytes.decode())
    state = checkpoint_state(root_path)
    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_seed_hex))
    body = {
        "schema": CHECKPOINT_SIG_SCHEMA,
        "algorithm": "ed25519",
        "key_id": key_id(pubkey_hex),
        "payload": state,
        "signature": key.sign(canonical_json_bytes(state)).hex(),
    }
    atomic_write_text(root_path / path, json.dumps(body, indent=2, sort_keys=True) + "\n")
    return root_path / path


def anchor_checkpoint(root: str | Path) -> Path:
    """RFC 3161-anchor the checkpoint — one TSA token time-binds the pins."""
    return stamp_timestamp(DEFAULT_CHECKPOINT_PATH, root=root)


def verify_checkpoint(root: str | Path) -> dict[str, Any]:
    """Verify the committed checkpoint against the committed pubkey + anchors.

    ``ok`` = authentic: signature verifies under ``gate_signing.pub``, the
    payload is well-formed, and the TSA anchor commits to this checkpoint
    file. ``current`` = the pinned digests still match the live tree —
    False means the pin state moved on (expected after ``stamp-epochs``),
    which deprecates but never invalidates the checkpoint.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    root_path = Path(root)
    errors: list[str] = []
    cp_rel = DEFAULT_CHECKPOINT_PATH.as_posix()
    cp_file = root_path / DEFAULT_CHECKPOINT_PATH
    pub_file = root_path / DEFAULT_PUBKEY_PATH
    if not cp_file.exists():
        return {"ok": True, "signed": False, "current": False, "errors": []}
    if not pub_file.exists():
        return {"ok": False, "signed": True, "current": False, "errors": ["pubkey_missing"]}
    try:
        body = json.loads(cp_file.read_text())
    except json.JSONDecodeError:
        return {"ok": False, "signed": True, "current": False, "errors": ["checkpoint_malformed"]}
    payload = body.get("payload")
    if (
        body.get("algorithm") != "ed25519"
        or body.get("schema") != CHECKPOINT_SIG_SCHEMA
        or not isinstance(payload, dict)
        or payload.get("schema") != CHECKPOINT_SCHEMA
    ):
        return {"ok": False, "signed": True, "current": False, "errors": ["checkpoint_malformed"]}
    pubkey_hex = pub_file.read_text().strip()
    try:
        pubkey = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
    except ValueError:
        return {"ok": False, "signed": True, "current": False, "errors": ["pubkey_malformed"]}
    if body.get("key_id") != key_id(pubkey_hex):
        errors.append("key_id_mismatch")
    try:
        pubkey.verify(bytes.fromhex(str(body.get("signature", ""))), canonical_json_bytes(payload))
    except (InvalidSignature, ValueError):
        errors.append("signature_invalid")

    # Currency: pinned digests vs live bytes.
    pins = payload.get("pins", {})
    current = True
    if isinstance(pins, dict):
        for rel in PINNED_FILES:
            declared = pins.get(rel)
            member = root_path / rel
            if (
                declared is None
                or not member.exists()
                or hash_bytes(member.read_bytes()) != declared
            ):
                current = False
    else:
        current = False

    # Authenticity of *when*: the anchor must commit to this checkpoint file.
    ts = verify_timestamps(root_path)
    anchored = False
    if ts.get("anchored"):
        anchored = bool(ts.get("fresh", {}).get(cp_rel))
        for err in ts.get("errors", []):
            if cp_rel in str(err) or "checkpoint" in str(err):
                errors.append(f"anchor:{err}")
    # No anchor at all: neutral (unsigned trees never reach here — signed only).

    # Predecessor continuity: when witness proofs exist, the declared
    # prev_sha256 must be one of the publicly witnessed digests — a
    # fabricated or rolled-back predecessor can't satisfy this without
    # having been committed to Rekor first.
    prev = payload.get("prev_sha256")
    if prev is not None:
        from quant_fund.research.integrity_witness import WITNESS_DIR

        witnessed: set[str] = set()
        for p in (root_path / WITNESS_DIR).glob(f"{DEFAULT_CHECKPOINT_PATH.name}_*.json"):
            try:
                digest = json.loads(p.read_text()).get("target", {}).get("sha256")
            except (OSError, json.JSONDecodeError, AttributeError):
                continue
            if isinstance(digest, str):
                witnessed.add(digest)
        if witnessed and prev not in witnessed:
            errors.append("prev_not_witnessed")
        # Local walk: if the predecessor's bytes were archived, its digest
        # must match prev exactly (defense in depth for the unwitnessed case).
        archive = root_path / CHECKPOINT_ARCHIVE_DIR
        if archive.is_dir():
            found = any(hash_bytes(f.read_bytes()) == prev for f in archive.glob("*.json"))
            if not found and prev != hash_bytes(cp_file.read_bytes()):
                errors.append("prev_not_archived")

    return {
        "ok": not errors,
        "signed": True,
        "anchored": anchored,
        "current": current,
        "errors": errors,
    }


def checkpoint_contract_errors(payload: Any) -> list[str]:
    """Lane contract for the checkpoint *state* payload."""
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        return ["schema_mismatch"]
    errors: list[str] = []
    pins = payload.get("pins")
    if not isinstance(pins, dict):
        errors.append("pins_missing")
    else:
        for rel in PINNED_FILES:
            digest = pins.get(rel)
            if not isinstance(digest, str) or len(digest) != 64:
                errors.append(f"pin_digest_malformed:{rel}")
    heads = payload.get("heads")
    if not isinstance(heads, dict) or not heads:
        errors.append("heads_missing")
    else:
        for key, entry in heads.items():
            if not isinstance(entry, dict) or not entry.get("receipt"):
                errors.append(f"head_entry_malformed:{key}")
    prev = payload.get("prev_sha256")
    if prev is not None and (not isinstance(prev, str) or len(prev) != 64):
        errors.append("prev_sha256_malformed")
    code = payload.get("code")
    if code is not None:
        if not isinstance(code, dict):
            errors.append("code_not_mapping")
        else:
            for field, size in (("revision", 40), ("worktree_sha256", 64)):
                v = code.get(field)
                if v == "UNKNOWN":
                    continue
                if not isinstance(v, str) or len(v) != size:
                    errors.append(f"code_{field}_malformed")
    return errors
