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
from quant_fund.research.gate_signatures import (
    DEFAULT_PUBKEY_PATH,
    DEFAULT_QUORUM_PATH,
    key_id,
    load_quorum_registry,
)
from quant_fund.research.timestamp_anchor import stamp_timestamp, verify_timestamps
from quant_fund.utils.atomicio import atomic_write_text
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

CHECKPOINT_SCHEMA = "integrity_checkpoint.v1"
CHECKPOINT_SIG_SCHEMA = "integrity_checkpoint_sig.v1"
# v2 wraps the same payload in an M-of-N signature list resolved against the
# committed gate_quorum.v1 registry — the checkpoint stops depending on a
# single signing key once a quorum registry exists.
CHECKPOINT_SIG_SCHEMA_V2 = "integrity_checkpoint_sig.v2"
DEFAULT_CHECKPOINT_PATH = Path("quality/checkpoint.json")
# Superseded checkpoints are archived append-only under this dir so the
# prev_sha256 chain is locally verifiable — not just digest-referenced via
# the Rekor proofs. The epoch corpus exempts this prefix (like witness/):
# archiving must not force a re-stamp on every checkpoint rotation.
CHECKPOINT_ARCHIVE_DIR = Path("quality/checkpoints")

# The pin files whose bytes the checkpoint covers. gate_pins.sig is included
# so the checkpoint also binds *which* pin signature was current, and
# gate_quorum.json so a registry rotation deprecates the stale head.
PINNED_FILES = (
    "quality/epoch_heads.json",
    "quality/crown_jewels.json",
    "quality/gate_quorum.json",
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
    # Spine extent: the archive set itself is corpus-exempt (the spine gate
    # covers it), so the checkpoint pins its count and tip — truncating an
    # archive record would otherwise leave a self-consistent shorter chain.
    archive_dir = root_path / CHECKPOINT_ARCHIVE_DIR
    archives = sorted(p.name for p in archive_dir.glob("*.json")) if archive_dir.is_dir() else []
    # Witness extent: the proof for THIS checkpoint lands after it is
    # written, so the verifier accepts exactly the claimed set or that
    # set plus the accreted live proof — deleting any older proof leaves
    # a claimed name missing.
    from quant_fund.research.integrity_witness import WITNESS_DIR

    wdir = root_path / WITNESS_DIR
    proofs = (
        {
            p.name: hash_bytes(p.read_bytes())
            for p in wdir.glob(f"{DEFAULT_CHECKPOINT_PATH.name}_*.json")
        }
        if wdir.is_dir()
        else {}
    )
    # Quorum binding: v2 checkpoints are authorized under a registry; the
    # signed payload names which one — canon-digest, threshold, key count —
    # so a swapped-in registry can't launder forged signatures. ``None``
    # records the pre-registry era explicitly.
    from quant_fund.research.gate_signatures import quorum_registry_errors, registry_sha256

    quorum_file = root_path / DEFAULT_QUORUM_PATH
    quorum_field: dict[str, Any] | None = None
    if quorum_file.is_file():
        raw_q = quorum_file.read_bytes()
        try:
            registry = json.loads(raw_q)
            qerrs = quorum_registry_errors(registry)
            quorum_field = {
                "registry_sha256": registry_sha256(registry) if not qerrs else hash_bytes(raw_q),
                "threshold": int(registry["threshold"]) if not qerrs else None,
                "n_keys": len(registry["keys"]) if not qerrs else None,
            }
            if qerrs:
                quorum_field["malformed"] = True
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            quorum_field = {"registry_sha256": hash_bytes(raw_q), "malformed": True}

    from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256

    return {
        "schema": CHECKPOINT_SCHEMA,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "quorum": quorum_field,
        "pins": pins,
        "heads": heads,
        "missing_pins": missing,
        "prev_sha256": prev_sha256,
        "spine": {
            "n_archives": len(archives),
            "tip": archives[-1] if archives else None,
        },
        "witness": {
            "n_proofs": len(proofs),
            "proofs": proofs,
        },
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
    signers: list[tuple[str, str]],
    *,
    path: Path = DEFAULT_CHECKPOINT_PATH,
) -> Path:
    """Sign ``checkpoint_state`` and write the checkpoint file atomically.

    ``signers`` are ``(private_seed_hex, pubkey_hex)`` pairs. When a clean
    ``gate_quorum.v1`` registry is committed the checkpoint is emitted as
    ``integrity_checkpoint_sig.v2`` — every signer must be registered and a
    below-quorum signature set refuses to write (a partial checkpoint is a
    brick, not evidence). Pre-quorum trees without a registry keep the
    single-signer ``v1`` envelope, signed by ``signers[0]``.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if not signers:
        raise ValueError("no signer key material")
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
    msg = canonical_json_bytes(state)
    quorum = load_quorum_registry(root_path)
    if quorum is not None:
        registered, threshold = quorum
        signatures: list[dict[str, str]] = []
        seen: set[str] = set()
        for priv_hex, pub_hex in signers:
            kid = key_id(pub_hex)
            if kid not in registered or registered[kid] != pub_hex:
                raise ValueError(f"signer not in quorum registry: {kid}")
            if kid in seen:
                continue
            seen.add(kid)
            sig = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex)).sign(msg)
            signatures.append({"key_id": kid, "signature": sig.hex()})
        if len(signatures) < threshold:
            raise ValueError(
                f"checkpoint quorum unattainable: {len(signatures)} signers < threshold {threshold}"
            )
        body: dict[str, Any] = {
            "schema": CHECKPOINT_SIG_SCHEMA_V2,
            "algorithm": "ed25519",
            "signatures": signatures,
            "payload": state,
        }
    else:
        priv_hex, pub_hex = signers[0]
        key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex))
        body = {
            "schema": CHECKPOINT_SIG_SCHEMA,
            "algorithm": "ed25519",
            "key_id": key_id(pub_hex),
            "payload": state,
            "signature": key.sign(msg).hex(),
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
        # Neutral only on trees that never checkpointed — an archive
        # record or witness proof means the layer was set up and the
        # checkpoint was deleted out from under it. (Pins-signed,
        # no-checkpoint trees are legal.)
        archive_dir = root_path / CHECKPOINT_ARCHIVE_DIR
        has_archive = archive_dir.is_dir() and any(archive_dir.glob("*.json"))
        from quant_fund.research.integrity_witness import WITNESS_DIR

        wdir = root_path / WITNESS_DIR
        has_witness = wdir.is_dir() and any(wdir.glob(f"{DEFAULT_CHECKPOINT_PATH.name}_*.json"))
        if has_archive or has_witness:
            return {
                "ok": False,
                "signed": True,
                "current": False,
                "errors": ["checkpoint_absent"],
            }
        return {"ok": True, "signed": False, "current": False, "errors": []}
    try:
        body = json.loads(cp_file.read_text())
    except (OSError, ValueError):
        return {"ok": False, "signed": True, "current": False, "errors": ["checkpoint_malformed"]}
    payload = body.get("payload")
    if (
        body.get("algorithm") != "ed25519"
        or body.get("schema") not in (CHECKPOINT_SIG_SCHEMA, CHECKPOINT_SIG_SCHEMA_V2)
        or not isinstance(payload, dict)
        or payload.get("schema") != CHECKPOINT_SCHEMA
    ):
        return {"ok": False, "signed": True, "current": False, "errors": ["checkpoint_malformed"]}
    # A committed quorum registry retires the single-key envelope at the head:
    # the checkpoint must prove the quorum, not one key. Registry absent or
    # malformed means nothing can satisfy it — v1 heads fail closed too. And
    # the *signed payload* names the authorizing registry's canonical digest:
    # the file on disk must be that registry, else the swap is silent.
    registry_present = (root_path / DEFAULT_QUORUM_PATH).exists()
    quorum = load_quorum_registry(root_path)
    if registry_present and quorum is None:
        errors.append("quorum_registry_malformed")
    if registry_present:
        claimed_q = payload.get("quorum")
        claimed_digest = claimed_q.get("registry_sha256") if isinstance(claimed_q, dict) else None
        if body.get("schema") == CHECKPOINT_SIG_SCHEMA_V2:
            if claimed_digest is None:
                errors.append("quorum_unbound")
            else:
                raw_disk = (root_path / DEFAULT_QUORUM_PATH).read_bytes()
                try:
                    from quant_fund.research.gate_signatures import registry_sha256

                    disk_registry = json.loads(raw_disk)
                    disk_digest = (
                        registry_sha256(disk_registry)
                        if quorum is not None
                        else hash_bytes(raw_disk)
                    )
                except (OSError, json.JSONDecodeError):
                    disk_digest = hash_bytes(raw_disk)
                if claimed_digest != disk_digest:
                    errors.append("quorum_registry_drift")
        registered, threshold = quorum if quorum is not None else ({}, 10**9)
        if body.get("schema") != CHECKPOINT_SIG_SCHEMA_V2:
            errors.append("checkpoint_below_quorum")
        else:
            sigs = body.get("signatures")
            if not isinstance(sigs, list) or not sigs:
                errors.append("checkpoint_below_quorum")
            else:
                valid: set[str] = set()
                for entry in sigs:
                    if not isinstance(entry, dict):
                        errors.append("signature_malformed")
                        continue
                    kid = entry.get("key_id")
                    sig_hex = entry.get("signature")
                    if kid not in registered:
                        errors.append("signature_key_unknown")
                        continue
                    if not isinstance(sig_hex, str):
                        errors.append("signature_malformed")
                        continue
                    try:
                        Ed25519PublicKey.from_public_bytes(
                            bytes.fromhex(registered[str(kid)])
                        ).verify(bytes.fromhex(sig_hex), canonical_json_bytes(payload))
                        valid.add(str(kid))
                    except (InvalidSignature, ValueError):
                        errors.append("signature_invalid")
                if len(valid) < threshold:
                    errors.append(f"checkpoint_below_quorum:{len(valid)}/{threshold}")
    elif body.get("schema") == CHECKPOINT_SIG_SCHEMA_V2:
        # v2 without a registry file: signers cannot be resolved.
        errors.append("quorum_registry_missing")
    else:
        if not pub_file.exists():
            return {"ok": False, "signed": True, "current": False, "errors": ["pubkey_missing"]}
        pubkey_hex = pub_file.read_text().strip()
        try:
            pubkey = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
        except ValueError:
            return {"ok": False, "signed": True, "current": False, "errors": ["pubkey_malformed"]}
        if body.get("key_id") != key_id(pubkey_hex):
            errors.append("key_id_mismatch")
        try:
            pubkey.verify(
                bytes.fromhex(str(body.get("signature", ""))), canonical_json_bytes(payload)
            )
        except (InvalidSignature, ValueError):
            errors.append("signature_invalid")

    # Currency: pinned digests vs live bytes. A file that is absent from
    # both the claim and the tree (a pre-registry-era v1 payload) is neutral.
    pins = payload.get("pins", {})
    current = True
    if isinstance(pins, dict):
        for rel in PINNED_FILES:
            declared = pins.get(rel)
            member = root_path / rel
            if declared is None and not member.exists():
                continue
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
            except (OSError, ValueError, AttributeError):
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

    # Provenance metadata, not a verdict: whether the checkpoint's claimed
    # code revision is an ancestor of this checkout's HEAD. A transplanted
    # checkpoint records a revision unrelated to this history — surfaced as
    # ``revision_ancestor=False`` without failing the gate, since a rebase
    # legitimately orphans the recorded sha (post-rebase checkpoints record
    # the new history again).
    code = payload.get("code")
    claimed_rev = code.get("revision") if isinstance(code, dict) else None
    revision_ancestor: bool | None = None
    if isinstance(claimed_rev, str) and len(claimed_rev) == 40 and (root_path / ".git").exists():
        try:
            import subprocess

            proc = subprocess.run(
                ["git", "merge-base", "--is-ancestor", claimed_rev, "HEAD"],
                cwd=root_path,
                capture_output=True,
                timeout=10,
            )
            revision_ancestor = proc.returncode == 0
        except (OSError, subprocess.SubprocessError):
            revision_ancestor = None

    return {
        "ok": not errors,
        "signed": True,
        "anchored": anchored,
        "current": current,
        "revision_ancestor": revision_ancestor,
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
        pins = {}
    else:
        for rel, digest in pins.items():
            if rel not in PINNED_FILES:
                errors.append(f"pin_unpinned:{rel}")
            elif not isinstance(digest, str) or len(digest) != 64:
                errors.append(f"pin_digest_malformed:{rel}")
    # A pinned file is either digested or declared missing — never absent
    # from both (a silent drop) nor claimed missing for a non-pin name.
    missing = payload.get("missing_pins")
    if not isinstance(missing, list):
        errors.append("missing_pins_malformed")
        missing = []
    missing_set = {str(m) for m in missing}
    for rel in PINNED_FILES:
        in_pins = rel in pins
        in_missing = rel in missing_set
        if in_pins == in_missing:
            errors.append(f"pin_accounting_error:{rel}")
    for name in sorted(missing_set - set(PINNED_FILES)):
        errors.append(f"missing_pins_nonmember:{name}")
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
