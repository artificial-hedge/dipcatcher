"""witness-scan — the transparency log as a key-misuse oracle.

Rekor witnessing (``integrity_witness``) proves our checkpoints reached the
log; this lane asks the *converse* question: has our witness key signed
anything **not** on the committed spine? A stolen ``WITNESS_SIGNING_KEY``
can mint a divergent checkpoint tree anywhere on earth — the attacker's
entries still land in Rekor under our key, and the log betrays them.

Protocol: ``POST /api/v1/index/retrieve`` with our committed public key
returns every entry UUID it ever signed; each entry's body carries the
sha256 we attested (hashedrekord ``data.hash.value``). A digest absent from
the committed checkpoint spine is ``foreign`` — cryptographic evidence of
key misuse that no repo access can hide.

Honesty contract: this is an online audit lane (like ``verify-witness
--online``) — Rekor unreachable reports ``unreachable`` and stays neutral,
never green-washed; malformed entries are ``unrecognized`` (informational,
not an accusation: only hashedrekord digests we recognize get classified).
"""

from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_REKOR_URL = "https://rekor.sigstore.dev"
WITNESS_PUBKEY_PATH = Path("quality/witness_signing.pub")
# Ceiling on entries fetched per scan — Rekor index can return thousands for
# a reused key; ours signs only checkpoints, so large scans are the anomaly.
MAX_ENTRIES = 512


def _http(url: str, timeout: float, data: bytes | None = None) -> bytes:
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  # nosec B310
        return bytes(resp.read())


def _entry_uuids(pubkey_pem: bytes, rekor_url: str, timeout: float) -> list[str]:
    """All Rekor entry UUIDs signed by ``pubkey_pem`` (may be capped)."""
    body = json.dumps(
        {
            "publicKey": {
                "format": "x509",
                "content": base64.b64encode(pubkey_pem).decode(),
            }
        }
    ).encode()
    raw = _http(rekor_url.rstrip("/") + "/api/v1/index/retrieve", timeout, body)
    uuids = json.loads(raw)
    if not isinstance(uuids, list):
        raise ValueError("index payload not a list")
    return [u for u in uuids if isinstance(u, str)][: MAX_ENTRIES + 1]


def _entry_digest(entry_resp: Any, uuid: str) -> str | None:
    """sha256 attested by a hashedrekord entry, or None if unrecognizable."""
    if not isinstance(entry_resp, dict):
        return None
    entry = entry_resp.get(uuid)
    if not isinstance(entry, dict):
        return None
    body_b64 = entry.get("body")
    if not isinstance(body_b64, str):
        return None
    try:
        spec = json.loads(base64.b64decode(body_b64))
        h = spec["spec"]["data"]["hash"]
        value = h["value"]
    except (ValueError, KeyError, TypeError):
        return None
    if (
        h.get("algorithm") != "sha256"
        or not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        return None
    return value


def _spine_digests(root: Path) -> set[str]:
    """Every checkpoint digest our committed spine claims — live + archives."""
    from quant_fund.utils.hashing import hash_bytes

    digests: set[str] = set()
    live = root / "quality/checkpoint.json"
    if live.is_file():
        digests.add(hash_bytes(live.read_bytes()))
    for p in (root / "quality/checkpoints").glob("*.json"):
        if p.is_file():
            digests.add(hash_bytes(p.read_bytes()))
    # Digests recorded inside committed witness proofs count too — a proof's
    # target.sha256 pins the checkpoint bytes at witness time even when the
    # checkpoint record itself predates the archive era.
    for p in (root / "quality/witness").glob("*.json"):
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        t = rec.get("target") if isinstance(rec, dict) else None
        sha = t.get("sha256") if isinstance(t, dict) else None
        if isinstance(sha, str) and len(sha) == 64:
            digests.add(sha)
    return digests


ORPHAN_REGISTRY = Path("quality/witness_orphans.json")
ORPHAN_SCHEMA = "witness_orphans.v1"


def _orphan_registry(root: Path) -> dict[str, Any]:
    """Committed registry of acknowledged foreign attestations.

    Pre-spine-era checkpoints were witnessed and then superseded — their
    digests live in Rekor forever but no committed record claims them. The
    registry names each such digest byte-exact; it can explain but never
    launder — an unlisted foreign digest still fails the scan."""
    try:
        body = json.loads((root / ORPHAN_REGISTRY).read_text())
    except (OSError, ValueError):
        return {}
    orphans = body.get("orphans") if body.get("schema") == ORPHAN_SCHEMA else None
    return orphans if isinstance(orphans, dict) else {}


def scan_witness_log(
    root: str | Path,
    *,
    rekor_url: str = DEFAULT_REKOR_URL,
    pubkey_path: Path = WITNESS_PUBKEY_PATH,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Sweep Rekor for every entry signed by our witness key.

    Returns ``{ok, scanned, foreign, unrecognized, truncated, errors}``.
    ``foreign`` digests are checkpoint hashes our key signed that no
    committed spine record claims — key-misuse evidence. ``ok`` is False on
    any foreign hit, unreachable index, or malformed pubkey file.
    """
    root_path = Path(root)
    pub_file = root_path / pubkey_path
    if not pub_file.is_file():
        return {
            "ok": False,
            "scanned": 0,
            "foreign": [],
            "unrecognized": 0,
            "truncated": False,
            "errors": ["witness_pubkey_missing"],
            "explained": [],
        }
    try:
        uuids = _entry_uuids(pub_file.read_bytes(), rekor_url, timeout)
    except Exception as exc:  # noqa: BLE001 — online lane reports, not raises
        # Fail closed: an unreachable index is NOT a clean scan — a
        # key-misuse oracle reporting ok while blind waves through a
        # stolen-key fork exactly when it matters. `online` separates the
        # unreachable case from a real negative scan; the tail formula
        # (`not foreign and not errors`) already implies ok=False here.
        return {
            "ok": False,
            "online": False,
            "scanned": 0,
            "foreign": [],
            "explained": [],
            "unrecognized": 0,
            "truncated": False,
            "errors": [f"index_unreachable:{exc.__class__.__name__}"],
        }
    truncated = len(uuids) > MAX_ENTRIES
    uuids = uuids[:MAX_ENTRIES]
    known = _spine_digests(root_path)
    orphans = _orphan_registry(root_path)
    foreign: list[str] = []
    explained: list[str] = []
    unrecognized = 0
    scanned = 0
    errors: list[str] = []
    for uuid in uuids:
        try:
            resp_raw = _http(rekor_url.rstrip("/") + f"/api/v1/log/entries/{uuid}", timeout)
            digest = _entry_digest(json.loads(resp_raw), uuid)
        except Exception as exc:  # noqa: BLE001 — a single dead entry skips
            errors.append(f"entry_fetch:{uuid[:12]}:{exc.__class__.__name__}")
            continue
        scanned += 1
        if digest is None:
            unrecognized += 1
        elif digest not in known:
            (explained if digest in orphans else foreign).append(digest)
    return {
        "ok": not foreign and not errors,
        "online": True,
        "scanned": scanned,
        "foreign": sorted(set(foreign)),
        "explained": sorted(set(explained)),
        "unrecognized": unrecognized,
        "truncated": truncated,
        "errors": sorted(errors),
    }
