"""Public transparency-log witnessing for the integrity checkpoint.

``timestamp_anchor`` proves *when* the pin state existed; ``integrity_witness``
proves it was committed to a **public append-only log** — Rekor, sigstore's
transparency log — which not even the repo owners can rewrite or retract.
The submitted artifact is a ``hashedrekord`` entry over
``quality/checkpoint.json``: its sha256 plus an ECDSA P-256 witness
signature (Rekor's hashedrekord requires SHA-256 + ECDSA/Ed25519ph; the
gate-signing Ed25519 key is Ed25519ph-incompatible there, so witnessing uses
a dedicated ECDSA keypair — the checkpoint sig itself stays Ed25519).

The committed proof under ``quality/witness/`` is fully self-verifying
**offline** — no network at verify time:

- the entry body's ``data.hash.value`` must equal the checkpoint's sha256
  (``current`` flag), and its embedded signature must verify under the
  committed ``quality/witness_signing.pub`` (proves *we* submitted it)
- the RFC 6962 inclusion proof walks the entry's merkle leaf
  (``sha256(0x00 || canonical body)``, embedded in the entry UUID) up the
  recorded audit path to ``rootHash``
- ``signedEntryTimestamp`` verifies under the pinned Rekor pubkey
  (ECDSA over the JCS-canonical ``LogEntryAnon``) — Rekor attests the
  entry existed at ``integratedTime``
- the checkpoint note's own signature verifies under the same pubkey
  (sig over ``payload + "\\n"``, 4-byte key-hint prefix stripped) — Rekor
  attests the root the proof resolves to

Missing proofs are neutral (``witnessed: False``), never a failure — the
public log is an optional outer layer, not a required gate.

Provenance evidence only; never a market or P&L claim.
"""

from __future__ import annotations

import base64
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from quant_fund.utils.atomicio import atomic_write_text
from quant_fund.utils.hashing import hash_bytes

WITNESS_SCHEMA = "integrity_witness.v1"
WITNESS_DIR = Path("quality/witness")
DEFAULT_TARGET = Path("quality/checkpoint.json")
WITNESS_PUBKEY_PATH = Path("quality/witness_signing.pub")
REKOR_PUBKEY_PATH = Path("quality/rekor_pubkey.pem")
DEFAULT_REKOR_URL = "https://rekor.sigstore.dev"


def _post_entry(target_bytes: bytes, witness_key_pem: bytes, rekor_url: str) -> dict[str, Any]:
    """Submit a hashedrekord entry for ``target_bytes``; return the log entry."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
        load_pem_private_key,
    )

    key = load_pem_private_key(witness_key_pem, password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(key.curve, ec.SECP256R1):
        raise ValueError("witness key must be an ECDSA P-256 PEM")
    digest = hashlib.sha256(target_bytes).hexdigest()
    sig = key.sign(target_bytes, ec.ECDSA(hashes.SHA256()))
    pub_pem = key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    body = {
        "kind": "hashedrekord",
        "apiVersion": "0.0.1",
        "spec": {
            "data": {"hash": {"algorithm": "sha256", "value": digest}},
            "signature": {
                "content": base64.b64encode(sig).decode(),
                "publicKey": {"content": base64.b64encode(pub_pem).decode()},
            },
        },
    }
    req = urllib.request.Request(
        rekor_url.rstrip("/") + "/api/v1/log/entries",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310  # nosec B310
        result: dict[str, Any] = json.loads(resp.read())
        return result


def submit_witness(
    root: str | Path,
    witness_key_pem: bytes,
    *,
    target: Path = DEFAULT_TARGET,
    rekor_url: str = DEFAULT_REKOR_URL,
) -> Path:
    """Witness ``target`` into Rekor and commit the self-verifying proof.

    Writes ``quality/witness/<target-stem>_<logIndex>.json``. The proof
    records everything an offline verifier needs; only the 32-byte digest
    and our ECDSA signature ever leave the machine.
    """
    root_path = Path(root)
    target_bytes = (root_path / target).read_bytes()
    entry_resp = _post_entry(target_bytes, witness_key_pem, rekor_url)
    record = _witness_record(target.as_posix(), target_bytes, entry_resp, rekor_url)
    out_dir = root_path / WITNESS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{target.name}_{record['rekor']['log_index']}.json"
    atomic_write_text(out, json.dumps(record, indent=2, sort_keys=True) + "\n")
    return out


def witness_bytes(
    target_file: str,
    target_bytes: bytes,
    witness_key_pem: bytes,
    *,
    rekor_url: str = DEFAULT_REKOR_URL,
    rekor_pubkey_pem: bytes | None = None,
) -> dict[str, Any]:
    """Witness arbitrary bytes into Rekor; return the proof record.

    Unlike ``submit_witness`` (which reads a repo file and writes the proof
    under ``quality/witness/``), this returns the record for the caller to
    embed — e.g. inside a ``release_attestation.v1`` payload. Both pubkeys
    are embedded in the record so the proof verifies with no repo access;
    ``rekor_pubkey_pem`` should be the committed pinned key.
    """
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
        load_pem_private_key,
    )

    key = load_pem_private_key(witness_key_pem, password=None)
    witness_pub_pem = key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    entry_resp = _post_entry(target_bytes, witness_key_pem, rekor_url)
    return _witness_record(
        target_file,
        target_bytes,
        entry_resp,
        rekor_url,
        witness_pubkey_pem=witness_pub_pem,
        rekor_pubkey_pem=rekor_pubkey_pem,
    )


def _witness_record(
    target_file: str,
    target_bytes: bytes,
    entry_resp: dict[str, Any],
    rekor_url: str,
    *,
    witness_pubkey_pem: bytes | None = None,
    rekor_pubkey_pem: bytes | None = None,
) -> dict[str, Any]:
    """Shape a Rekor entry response into a self-verifying proof record.

    ``keys`` (only present when pubkeys are supplied) embeds the pubkeys the
    record verifies under — for proofs meant to travel outside the repo
    (embedded attestation witnesses), so an auditor needs no repo at all.
    """
    uuid, entry = next(iter(entry_resp.items()))
    ver = entry.get("verification", {})
    proof = ver.get("inclusionProof", {})
    record: dict[str, Any] = {
        "schema": WITNESS_SCHEMA,
        "target": {"file": target_file, "sha256": hash_bytes(target_bytes)},
        "rekor": {
            "url": rekor_url,
            "uuid": uuid,
            "log_id": entry.get("logID"),
            "log_index": entry.get("logIndex"),
            "integrated_time": entry.get("integratedTime"),
            "body_b64": entry.get("body"),
            "signed_entry_timestamp": ver.get("signedEntryTimestamp"),
            "inclusion_proof": {
                "log_index": proof.get("logIndex"),
                "tree_size": proof.get("treeSize"),
                "root_hash": proof.get("rootHash"),
                "hashes": proof.get("hashes", []),
                "checkpoint_note": proof.get("checkpoint"),
            },
        },
    }
    if witness_pubkey_pem is not None or rekor_pubkey_pem is not None:
        record["keys"] = {
            "witness_pubkey_pem": witness_pubkey_pem.decode() if witness_pubkey_pem else None,
            "rekor_pubkey_pem": rekor_pubkey_pem.decode() if rekor_pubkey_pem else None,
        }
    return record


def _ecdsa_verify(pubkey_pem: bytes, sig_b64: str, message: bytes) -> bool:
    """True iff DER ECDSA/SHA-256 ``sig_b64`` verifies over ``message``."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    try:
        pub = load_pem_public_key(pubkey_pem)
        if not isinstance(pub, ec.EllipticCurvePublicKey):
            return False
        pub.verify(base64.b64decode(sig_b64), message, ec.ECDSA(hashes.SHA256()))
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


def _merkle_leaf(body_b64: str) -> bytes:
    """Trillian/RFC 6962 leaf hash — also the tail of the entry UUID."""
    return hashlib.sha256(b"\x00" + base64.b64decode(body_b64)).digest()


def _audit_walk(leaf: bytes, log_index: int, tree_size: int, hashes: list[str]) -> str:
    """RFC 6962 inclusion walk: leaf + audit path -> reconstructed root."""
    fn, sn = log_index, tree_size - 1
    r = leaf
    for h in hashes:
        hb = bytes.fromhex(h)
        if (fn & 1) or fn == sn:
            r = hashlib.sha256(b"\x01" + hb + r).digest()
            while fn and not (fn & 1):
                fn >>= 1
                sn >>= 1
        else:
            r = hashlib.sha256(b"\x01" + r + hb).digest()
        fn >>= 1
        sn >>= 1
    return r.hex()


def verify_witness_file(root: str | Path, proof_path: str | Path) -> dict[str, Any]:
    """Verify one committed witness proof fully offline.

    ``ok`` = the proof is authentic: target digest re-derived from the entry
    body, our witness signature over the artifact verifies, the RFC 6962
    path reaches the recorded root, and Rekor's SET + checkpoint-note
    signatures verify under the pinned Rekor pubkey. ``current`` = the
    target file's live bytes still hash to the witnessed digest.
    """
    root_path = Path(root)
    proof_file = root_path / proof_path
    try:
        record = json.loads(proof_file.read_text())
    except (OSError, ValueError):
        return {"ok": False, "errors": ["proof_malformed"]}
    target_file = root_path / str(record.get("target", {}).get("file", ""))
    target_bytes = target_file.read_bytes() if target_file.is_file() else None
    pub_file = root_path / WITNESS_PUBKEY_PATH
    rekor_pub = root_path / REKOR_PUBKEY_PATH
    res = verify_witness_record(
        record,
        target_bytes=target_bytes,
        witness_pubkey_pem=pub_file.read_bytes() if pub_file.is_file() else None,
        rekor_pubkey_pem=rekor_pub.read_bytes() if rekor_pub.is_file() else None,
    )
    res["current"] = bool(target_file.is_file() and res["current"])
    return res


def verify_witness_record(
    record: Any,
    *,
    target_bytes: bytes | None,
    witness_pubkey_pem: bytes | None,
    rekor_pubkey_pem: bytes | None,
) -> dict[str, Any]:
    """Verify a witness proof against caller-supplied bytes and pubkeys.

    Same checks as ``verify_witness_file`` but takes the witnessed bytes and
    both keys directly — this is what embedded proofs (an attestation whose
    ``witness`` field carries its own Rekor record) verify against. When
    ``target_bytes`` is given and digests to the logged digest, the artifact
    signature IS re-verified (unlike the file variant, derived bytes are
    always available). ``current`` = supplied bytes hash to the logged
    digest (False when ``target_bytes`` is None or drifts).
    """
    if not isinstance(record, dict):
        return {"ok": False, "current": False, "errors": ["proof_malformed"]}
    errors: list[str] = []
    if record.get("schema") != WITNESS_SCHEMA:
        errors.append("schema_mismatch")
    rekor = record.get("rekor", {})
    target = record.get("target", {})
    body_b64 = rekor.get("body_b64")
    if not isinstance(body_b64, str):
        return {"ok": False, "current": False, "errors": errors + ["body_missing"]}
    try:
        body = json.loads(base64.b64decode(body_b64))
    except (ValueError, json.JSONDecodeError):
        return {"ok": False, "current": False, "errors": errors + ["body_malformed"]}

    # The logged digest is the target's — and we signed it (attribution).
    spec = body.get("spec", {})
    logged_digest = spec.get("data", {}).get("hash", {}).get("value", "")
    if logged_digest != target.get("sha256"):
        errors.append("digest_mismatch")
    current = target_bytes is not None and hash_bytes(target_bytes) == logged_digest
    if witness_pubkey_pem is None:
        errors.append("witness_pubkey_missing")
    elif (
        current
        and isinstance(spec.get("signature"), dict)
        # The logged signature covers the ORIGINAL artifact bytes — only
        # re-verifiable while the supplied bytes still match the digest.
        and not _ecdsa_verify(
            witness_pubkey_pem,
            spec["signature"].get("content", ""),
            target_bytes or b"",
        )
    ):
        errors.append("witness_signature_invalid")
    # Self-authentication for embedded proofs: when the record carries its
    # own witness pubkey, it must equal the one the LOG recorded — the log
    # entry (itself SET-signed + merkle-anchored) is the key's root of trust.
    embedded = record.get("keys", {})
    if isinstance(embedded, dict) and embedded.get("witness_pubkey_pem") is not None:
        logged_key = (
            spec.get("signature", {}).get("publicKey", {}).get("content", "")
            if isinstance(spec.get("signature"), dict)
            else ""
        )
        # logged content is base64(PEM); embedded stores raw PEM text.
        if base64.b64encode(str(embedded["witness_pubkey_pem"]).encode()).decode() != str(
            logged_key
        ):
            errors.append("witness_key_mismatch")

    # The UUID's tail IS the merkle leaf hash — binds entry to proof.
    uuid = str(rekor.get("uuid", ""))
    leaf = _merkle_leaf(body_b64)
    if not uuid.endswith(leaf.hex()):
        errors.append("leaf_uuid_mismatch")
    ip = rekor.get("inclusion_proof", {})
    try:
        root_hash = _audit_walk(
            leaf,
            int(ip["log_index"]),
            int(ip["tree_size"]),
            list(ip.get("hashes", [])),
        )
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "current": current, "errors": errors + ["inclusion_proof_malformed"]}
    if root_hash != ip.get("root_hash"):
        errors.append("inclusion_root_mismatch")

    if rekor_pubkey_pem is not None:
        # SET: Rekor's signature over the JCS-canonical LogEntryAnon.
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
        if not _ecdsa_verify(rekor_pubkey_pem, str(rekor.get("signed_entry_timestamp", "")), canon):
            errors.append("set_signature_invalid")
        note = str(ip.get("checkpoint_note", ""))
        payload, _, sig_block = note.partition("\n\n")
        if payload and sig_block:
            raw = base64.b64decode(sig_block.strip().split(" ")[-1])
            if not _ecdsa_verify(
                rekor_pubkey_pem, base64.b64encode(raw[4:]).decode(), (payload + "\n").encode()
            ):
                errors.append("checkpoint_note_signature_invalid")
        else:
            errors.append("checkpoint_note_malformed")
    else:
        errors.append("rekor_pubkey_missing")

    return {"ok": not errors, "current": current, "errors": sorted(errors)}


def _h(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _consistency_ok(
    old_size: int, old_root: bytes, new_size: int, new_root: bytes, path: list[bytes]
) -> bool:
    """RFC 6962 §2.1.4.2: ``old_root`` is a prefix-tree of ``new_root``.

    ``fr`` accumulates toward the new root, ``sr`` toward the old — the
    first path hash serves both walks (it is the old tree's "frontier" node).
    """
    if old_size == 0:
        return True
    if old_size == new_size:
        return old_root == new_root and not path
    if not path:
        return False
    fn, sn = old_size - 1, new_size - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1
    fr = sr = path[0]
    for c in path[1:]:
        if sn == 0:
            return False
        if (fn & 1) or fn == sn:
            fr = _h(c, fr)
            sr = _h(c, sr)
            while fn and not (fn & 1):
                fn >>= 1
                sn >>= 1
        else:
            fr = _h(fr, c)
        fn >>= 1
        sn >>= 1
    return fr == new_root and sr == old_root


def _parse_checkpoint_note(note: str) -> tuple[int, bytes] | None:
    """``size, rootHash`` from a signed checkpoint's payload text."""
    payload = note.split("\n\n")[0]
    lines = payload.split("\n")
    if len(lines) != 3:
        return None
    try:
        return int(lines[1]), base64.b64decode(lines[2])
    except (ValueError, TypeError):
        return None


def verify_witness_online(
    root: str | Path,
    *,
    rekor_url: str = DEFAULT_REKOR_URL,
    timeout: int = 30,
) -> dict[str, Any]:
    """Live-log check: is the committed proof's tree still *part of* Rekor's
    current signed tree head?

    The offline proofs bind our entry to the rootHash the log produced at
    submission time; this check proves the log still stands behind that
    state — i.e. the committed tree is a prefix of today's signed root (a
    dropped/rewritten segment breaks the RFC 6962 consistency path), and
    the current STH signature verifies under the pinned Rekor pubkey.
    """
    res = verify_witnesses(root)
    if not res["witnessed"]:
        return {"ok": True, "online": False, "witnessed": False, "errors": []}
    pub_file = Path(root) / REKOR_PUBKEY_PATH
    errors: list[str] = []
    try:
        with urllib.request.urlopen(  # noqa: S310  # nosec B310
            rekor_url.rstrip("/") + "/api/v1/log", timeout=timeout
        ) as resp:
            log = json.loads(resp.read())
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "online": False, "errors": [f"log_unreachable:{exc}"]}
    note = str(log.get("signedTreeHead", ""))
    parsed = _parse_checkpoint_note(note)
    if parsed is None:
        return {"ok": False, "online": True, "errors": ["sth_malformed"]}
    cur_size, cur_root = parsed
    if cur_root.hex() != str(log.get("rootHash", "")):
        errors.append("sth_root_mismatch")
    if pub_file.is_file():
        payload, _, sig_block = note.partition("\n\n")
        raw = base64.b64decode(sig_block.strip().split(" ")[-1]) if sig_block else b""
        if not sig_block or not _ecdsa_verify(
            pub_file.read_bytes(), base64.b64encode(raw[4:]).decode(), (payload + "\n").encode()
        ):
            errors.append("sth_signature_invalid")
    else:
        errors.append("rekor_pubkey_missing")

    proofs = res["proofs"]
    consistent: dict[str, bool] = {}
    for name, pres in proofs.items():
        ip = pres.get("inclusion_proof") or {}
        # proofs from verify_witness_file don't carry ip; reload the record
        try:
            record = json.loads((Path(root) / WITNESS_DIR / name).read_text())
            ip = record["rekor"]["inclusion_proof"]
        except (OSError, ValueError, KeyError, TypeError):
            errors.append(f"{name}:proof_malformed")
            consistent[name] = False
            continue
        old_size, old_root = int(ip["tree_size"]), bytes.fromhex(str(ip["root_hash"]))
        if old_size > cur_size:
            errors.append(f"{name}:log_shrunk")
            consistent[name] = False
            continue
        if old_size == cur_size:
            consistent[name] = old_root == cur_root
            if not consistent[name]:
                errors.append(f"{name}:root_diverged")
            continue
        try:
            with urllib.request.urlopen(  # noqa: S310  # nosec B310
                rekor_url.rstrip("/")
                + f"/api/v1/log/proof?firstSize={old_size}&lastSize={cur_size}",
                timeout=timeout,
            ) as resp:
                path = [bytes.fromhex(h) for h in json.loads(resp.read())["hashes"]]
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{name}:proof_fetch_failed:{exc}")
            continue
        ok = _consistency_ok(old_size, old_root, cur_size, cur_root, path)
        consistent[name] = ok
        if not ok:
            errors.append(f"{name}:consistency_proof_invalid")
    return {
        "ok": not errors,
        "online": True,
        "witnessed": True,
        "errors": sorted(errors),
        "consistent": consistent,
        "log_size": cur_size,
    }


def verify_witnesses(root: str | Path) -> dict[str, Any]:
    """All committed proofs under ``quality/witness/``; neutral when absent."""
    wdir = Path(root) / WITNESS_DIR
    if not wdir.is_dir() or not any(wdir.glob("*.json")):
        return {"ok": True, "witnessed": False, "errors": [], "proofs": {}}
    proofs: dict[str, Any] = {}
    errors: list[str] = []
    for pf in sorted(wdir.glob("*.json")):
        res = verify_witness_file(root, pf)
        proofs[pf.name] = res
        errors.extend(f"{pf.name}:{e}" for e in res["errors"])
    return {
        "ok": not errors,
        "witnessed": True,
        "errors": sorted(errors),
        "proofs": proofs,
    }


def witness_contract_errors(payload: Any) -> list[str]:
    """Lane contract for a committed ``integrity_witness.v1`` proof."""
    if not isinstance(payload, dict) or payload.get("schema") != WITNESS_SCHEMA:
        return ["schema_mismatch"]
    errors: list[str] = []
    target = payload.get("target")
    if not isinstance(target, dict) or len(str(target.get("sha256", ""))) != 64:
        errors.append("target_malformed")
    rekor = payload.get("rekor")
    if not isinstance(rekor, dict):
        errors.append("rekor_missing")
    else:
        for field in ("uuid", "log_id", "body_b64", "signed_entry_timestamp"):
            if not rekor.get(field):
                errors.append(f"rekor_field_missing:{field}")
        ip = rekor.get("inclusion_proof")
        if not isinstance(ip, dict) or not ip.get("hashes") or not ip.get("root_hash"):
            errors.append("inclusion_proof_missing")
    return errors
