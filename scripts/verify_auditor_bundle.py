#!/usr/bin/env python3
"""Standalone auditor-bundle verifier — deliberately independent.

Everything in ``quant_fund.research.auditor_bundle`` is verified code, but
a single verifier is itself a trust bottleneck: a subtle bug (or a silent
patch) in the repo's implementation could hide a forgery. This script is
the second implementation — it imports NOTHING from quant_fund, only the
stdlib + ``cryptography``. An auditor can read it top-to-bottom (~300
lines) and verify a bundle against the public Rekor log alone.

Usage:
    python scripts/verify_auditor_bundle.py auditor_bundle.json \
        [--rekor-pubkey rekor.pem] [--offline]

Exit code 0 = every link verified; 1 = any failure (each printed).

Chain verified (same semantics as verify_bundle, reimplemented):
  1. files_sha256 manifest over every b64 member
  2. Rekor inclusion: leaf = sha256(0x00 || entry-body-bytes); audit-path
     walk to the recorded rootHash; UUID tail must equal the leaf hash
  3. Rekor SET: ECDSA/SHA-256 over the JCS-canonical LogEntryAnon
     (sorted-key JSON of {body, integratedTime, logID, logIndex})
  4. Rekor checkpoint note: sig over payload+"\n" (4-byte key-hint prefix
     stripped from the note's signature blob)
  5. checkpoint sha256 == the digest inside the entry body
  6. witness signature over checkpoint bytes under the pubkey the LOG
     recorded as signer — which must equal the bundled witness pubkey
  7. checkpoint Ed25519 signature over its signed ``payload`` object
     (canonical JSON); checkpoint's ``pins`` must match the bundled pin
     files' sha256; crown_jewels must pin the gate pubkey's bytes;
     gate_pins.sig verifies over its own embedded payload whose declared
     digests must match the bundled pin files
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

REKOR_PUBLIC_KEY_URL = "https://rekor.sigstore.dev/api/v1/log/publicKey"

MEMBERS = (
    "quality/checkpoint.json",
    "quality/epoch_heads.json",
    "quality/crown_jewels.json",
    "quality/gate_signing.pub",
    "gate_pins.sig",
    "quality/witness_signing.pub",
    "quality/rekor_pubkey.pem",
)


def _sha(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _canon(value: Any) -> bytes:
    """canonical_json_bytes equivalent for plain JSON types."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _ecdsa_pem_verify(pem: bytes, sig_der: bytes, msg: bytes) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    try:
        pub = load_pem_public_key(pem)
        if not isinstance(pub, ec.EllipticCurvePublicKey):
            return False
        pub.verify(sig_der, msg, ec.ECDSA(hashes.SHA256()))
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


def _ed25519_verify(pub_hex: str, sig_hex: str, msg: bytes) -> bool:
    """Repo convention: Ed25519 keys/sigs are raw 32/64-byte hex, not PEM."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex.strip()))
        pub.verify(bytes.fromhex(sig_hex), msg)
    except (InvalidSignature, ValueError):
        return False
    return True


def _inclusion_walk(leaf: bytes, log_index: int, tree_size: int, path: list[bytes]) -> bytes:
    """RFC 6962 §2.1.1 audit path → reconstructed root."""
    fn, sn = log_index, tree_size - 1
    r = leaf
    for h in path:
        if (fn & 1) or fn == sn:
            r = _sha(b"\x01" + h + r)
            while fn and not (fn & 1):
                fn >>= 1
                sn >>= 1
        else:
            r = _sha(b"\x01" + r + h)
        fn >>= 1
        sn >>= 1
    return r


def verify(bundle_path: Path, rekor_pem: bytes | None) -> list[str]:
    errors: list[str] = []
    try:
        bundle = json.loads(bundle_path.read_text())
    except (OSError, json.JSONDecodeError):
        return ["bundle_unreadable"]
    if bundle.get("schema") != "auditor_bundle.v1":
        return ["schema_mismatch"]

    files: dict[str, str] = bundle.get("files", {})
    declared: dict[str, str] = bundle.get("files_sha256", {})
    proof: dict[str, Any] = bundle.get("witness_proof", {})
    decoded: dict[str, bytes] = {}
    for rel in MEMBERS:
        b64 = files.get(rel)
        if b64 is None:
            errors.append(f"missing:{rel}")
            continue
        try:
            raw = base64.b64decode(b64)
        except ValueError:
            errors.append(f"b64_malformed:{rel}")
            continue
        decoded[rel] = raw
        if declared.get(rel) != _sha(raw).hex():
            errors.append(f"files_sha256_mismatch:{rel}")
    if errors:
        return sorted(errors)

    rekor = proof.get("rekor", {})
    body_b64 = str(rekor.get("body_b64", ""))
    ip = rekor.get("inclusion_proof", {})
    if rekor_pem is None:
        rekor_pem = decoded["quality/rekor_pubkey.pem"]

    # -- Rekor entry -------------------------------------------------
    try:
        entry_body = json.loads(base64.b64decode(body_b64))
        spec = entry_body["spec"]
        logged_digest = spec["data"]["hash"]["value"]
        logged_sig = base64.b64decode(spec["signature"]["content"])
        logged_pub = base64.b64decode(spec["signature"]["publicKey"]["content"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        return errors + [f"entry_body_malformed:{exc}"]

    leaf = _sha(b"\x00" + base64.b64decode(body_b64))
    if not str(rekor.get("uuid", "")).endswith(leaf.hex()):
        errors.append("leaf_uuid_mismatch")
    try:
        root = _inclusion_walk(
            leaf,
            int(ip["log_index"]),
            int(ip["tree_size"]),
            [bytes.fromhex(h) for h in ip["hashes"]],
        )
        if root.hex() != ip["root_hash"]:
            errors.append("inclusion_root_mismatch")
    except (KeyError, TypeError, ValueError):
        errors.append("inclusion_proof_malformed")

    canon = json.dumps(
        {
            "body": body_b64,
            "integratedTime": rekor.get("integrated_time"),
            "logID": rekor.get("log_id"),
            "logIndex": rekor.get("log_index"),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    try:
        set_sig = base64.b64decode(str(rekor.get("signed_entry_timestamp", "")))
    except ValueError:
        set_sig = b""
    if not _ecdsa_pem_verify(rekor_pem, set_sig, canon):
        errors.append("set_signature_invalid")

    note = str(ip.get("checkpoint_note", ""))
    note_payload, _, sig_block = note.partition("\n\n")
    try:
        note_sig = base64.b64decode(sig_block.strip().split(" ")[-1])[4:]
    except (ValueError, IndexError):
        note_sig = b""
    if not sig_block or not _ecdsa_pem_verify(rekor_pem, note_sig, (note_payload + "\n").encode()):
        errors.append("checkpoint_note_signature_invalid")

    # -- our checkpoint ----------------------------------------------
    checkpoint_raw = decoded["quality/checkpoint.json"]
    if _sha(checkpoint_raw).hex() != logged_digest:
        errors.append("checkpoint_digest_mismatch")
    if not _ecdsa_pem_verify(logged_pub, logged_sig, checkpoint_raw):
        errors.append("witness_signature_invalid")
    if logged_pub != decoded["quality/witness_signing.pub"]:
        errors.append("witness_pubkey_diverges_from_log")

    gate_pub_hex = decoded["quality/gate_signing.pub"].decode().strip()
    try:
        checkpoint = json.loads(checkpoint_raw)
        payload = checkpoint["payload"]
        if not _ed25519_verify(gate_pub_hex, str(checkpoint["signature"]), _canon(payload)):
            errors.append("checkpoint_signature_invalid")
        pins = payload.get("pins", {})
        for rel in ("quality/crown_jewels.json", "quality/epoch_heads.json", "gate_pins.sig"):
            if pins.get(rel) != _sha(decoded[rel]).hex():
                errors.append(f"checkpoint_pin_drift:{rel}")
    except (KeyError, TypeError, json.JSONDecodeError):
        errors.append("checkpoint_malformed")

    # -- pins ---------------------------------------------------------
    try:
        crown = json.loads(decoded["quality/crown_jewels.json"])
        sigfile = json.loads(decoded["gate_pins.sig"])
    except json.JSONDecodeError:
        return sorted(set(errors + ["pins_malformed"]))

    crown_files = crown.get("files", {})
    if (
        crown_files.get("quality/gate_signing.pub")
        != _sha(decoded["quality/gate_signing.pub"]).hex()
    ):
        errors.append("gate_pubkey_not_pinned")

    sig_payload = sigfile.get("payload", {})
    sig_files = sig_payload.get("files", {}) if isinstance(sig_payload, dict) else {}
    for rel, want in sig_files.items():
        member = decoded.get(rel)
        if member is None:
            errors.append(f"pinned_file_missing:{rel}")
        elif _sha(member).hex() != want:
            errors.append(f"pin_drift:{rel}")
    if not _ed25519_verify(gate_pub_hex, str(sigfile.get("signature", "")), _canon(sig_payload)):
        errors.append("gate_pins_signature_invalid")

    return sorted(set(errors))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--rekor-pubkey", type=Path, default=None)
    ap.add_argument(
        "--offline",
        action="store_true",
        help="use the bundle's pinned Rekor key instead of fetching it live",
    )
    args = ap.parse_args()

    rekor_pem: bytes | None = None
    if args.rekor_pubkey is not None:
        rekor_pem = args.rekor_pubkey.read_bytes()
    elif not args.offline:
        try:
            with urllib.request.urlopen(  # noqa: S310
                REKOR_PUBLIC_KEY_URL, timeout=30
            ) as resp:
                rekor_pem = resp.read()
        except OSError as exc:
            print(f"warn: live key fetch failed ({exc}); using bundle pin", file=sys.stderr)

    errors = verify(args.bundle, rekor_pem)
    if errors:
        print("FAIL")
        for e in errors:
            print(f"  {e}")
        return 1
    print("ok — every link verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
