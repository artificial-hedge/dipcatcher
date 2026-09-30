"""Ed25519 signatures over the integrity pin files.

The sha256 pins (``quality/epoch_heads.json``, ``quality/crown_jewels.json``)
detect tamper, but a writer with tree access can re-pin after rewriting a
gate file. Signing the pin *manifest* with an Ed25519 key upgrades detection
to attribution: a forged pin needs the private key, and verification needs
only the committed public key.

Scope: the signature covers exactly the two pin files' *bytes* — one Ed25519
signature over ``canonical_json_bytes({"files": {path: sha256}})``. The
signature file lives at the repo root (``gate_pins.sig``), outside every
chained corpus dir, so stamping a chain never invalidates the signature it
attests (the stamp-order circularity).

Honesty contract: ``signed`` reports state, not truth — ``verify_pin_signatures``
returns ``unsigned`` when no pubkey/sig exists (a neutral verdict, not a
pass), ``ok`` only when both are present and verify.
"""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

GATE_SIGNATURES_SCHEMA = "gate_signatures.v1"
DEFAULT_PUBKEY_PATH = Path("quality/gate_signing.pub")
DEFAULT_SIG_PATH = Path("gate_pins.sig")
SIGNED_FILES = ("quality/epoch_heads.json", "quality/crown_jewels.json")


def _pin_manifest(root: Path) -> dict[str, dict[str, str]]:
    return {"files": {rel: hash_bytes((root / rel).read_bytes()) for rel in SIGNED_FILES}}


def manifest_bytes(root: Path) -> bytes:
    """Canonical signed payload — byte-exact for sign and verify."""
    return canonical_json_bytes(_pin_manifest(root))


def generate_keypair() -> tuple[str, str]:
    """Return (private_raw_seed_hex, public_raw_hex) — 32-byte seeds."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    key = Ed25519PrivateKey.generate()
    priv = key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    pub = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return priv.hex(), pub.hex()


def key_id(pubkey_hex: str) -> str:
    return hash_bytes(bytes.fromhex(pubkey_hex))[:16]


def sign_pins(
    root: str | Path,
    private_seed_hex: str,
    pubkey_hex: str,
) -> Path:
    """Write ``gate_pins.sig`` (and the pubkey file if absent). Atomic."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from quant_fund.utils.atomicio import atomic_write_text

    root_path = Path(root)
    manifest = _pin_manifest(root_path)
    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_seed_hex))
    sig = key.sign(canonical_json_bytes(manifest))
    body = {
        "schema": GATE_SIGNATURES_SCHEMA,
        "algorithm": "ed25519",
        "key_id": key_id(pubkey_hex),
        "payload": manifest,
        "signature": sig.hex(),
    }
    atomic_write_text(
        root_path / DEFAULT_PUBKEY_PATH,
        pubkey_hex + "\n",
    )
    sig_path = root_path / DEFAULT_SIG_PATH
    atomic_write_text(sig_path, json.dumps(body, indent=2, sort_keys=True) + "\n")
    return sig_path


def verify_pin_signatures(
    root: str | Path,
    *,
    pubkey_path: Path = DEFAULT_PUBKEY_PATH,
    sig_path: Path = DEFAULT_SIG_PATH,
) -> dict[str, object]:
    """Verify the pin manifest signature.

    ``errors`` is empty on a valid signature; ``signed`` distinguishes
    "unsigned repo" (no key material — neutral) from "verified" and "forged".
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    root_path = Path(root)
    errors: list[str] = []
    pub_file = root_path / pubkey_path
    sig_file = root_path / sig_path
    if not pub_file.exists() and not sig_file.exists():
        return {"ok": True, "signed": False, "errors": []}
    if not pub_file.exists():
        return {"ok": False, "signed": True, "errors": ["pubkey_missing"]}
    if not sig_file.exists():
        return {"ok": False, "signed": True, "errors": ["signature_file_missing"]}
    pubkey_hex = pub_file.read_text().strip()
    try:
        pubkey = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
    except ValueError:
        return {"ok": False, "signed": True, "errors": ["pubkey_malformed"]}
    try:
        body = json.loads(sig_file.read_text())
    except (OSError, ValueError):
        return {"ok": False, "signed": True, "errors": ["signature_file_malformed"]}
    if body.get("schema") != GATE_SIGNATURES_SCHEMA or body.get("algorithm") != "ed25519":
        return {"ok": False, "signed": True, "errors": ["signature_file_malformed"]}
    if body.get("key_id") != key_id(pubkey_hex):
        errors.append("key_id_mismatch")
    payload = body.get("payload", {})
    files = payload.get("files", {}) if isinstance(payload, dict) else {}
    if set(files) != set(SIGNED_FILES):
        errors.append("signed_file_set_drift")
    else:
        for rel, declared in files.items():
            member = root_path / rel
            if not member.exists():
                errors.append(f"pinned_file_missing:{rel}")
            elif hash_bytes(member.read_bytes()) != declared:
                errors.append(f"pin_drift:{rel}")
    if errors:
        return {"ok": False, "signed": True, "errors": sorted(errors)}
    try:
        pubkey.verify(bytes.fromhex(str(body.get("signature", ""))), canonical_json_bytes(payload))
    except InvalidSignature:
        errors.append("signature_invalid")
    except ValueError:
        errors.append("signature_malformed")
    return {"ok": not errors, "signed": True, "errors": sorted(errors)}
