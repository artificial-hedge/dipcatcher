"""RFC 3161 trusted-timestamp anchors over the integrity pins.

Seals prove a receipt's bytes are unchanged *since sealing*; epoch chains
prove corpus order. Neither proves *when* the state existed — a history
rewriter could mint the whole chain yesterday. A committed TSA token anchors
the pin bytes to an external timestamp authority's signature time: the
pinned state demonstrably existed by T.

Layout under ``quality/timestamps/``:

- ``anchors.json`` — ``timestamp_anchors.v1`` manifest mapping each
  ``<name>.tsr`` token to ``{target, sha256}`` (the digest the token must
  commit to, frozen at stamp time).
- ``<name>.tsr`` — DER ``TimeStampResp`` for that target.
- ``freetsa_cacert.pem`` / ``freetsa_tsa.crt`` — the TSA's CA + signer
  certs, committed so verification is offline and reproducible.

Implementation notes:

- ``build_tsq`` is a pure-Python DER ``TimeStampReq`` (no openssl, no extra
  dep): ``SEQUENCE { INTEGER 1, MessageImprint, BOOLEAN TRUE }``.
- ``stamp_timestamp`` POSTs via urllib (stdlib); the request carries only
  the 32-byte digest, never file contents, and stores the token only when
  its embedded imprint matches.
- ``extract_imprint`` walks the DER to TSTInfo's ``messageImprint`` in pure
  Python — no ASN.1 dependency.
- ``verify_timestamps`` checks the manifest↔token binding and each token's
  imprint against the live target (a mismatch means *stale anchor*, a valid
  proof of an earlier pin state — not tamper). Cert-chain verification uses
  ``openssl ts -verify`` when openssl + the committed certs exist; otherwise
  the chain is honestly reported ``chain_unchecked``.

Honesty contract: an unanchored repo is neutral, never a pass or a failure;
a malformed token or imprint/manifest mismatch is a hard error.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import hash_bytes

TIMESTAMPS_DIR = Path("quality/timestamps")
ANCHORS_MANIFEST = "anchors.json"
ANCHORS_SCHEMA = "timestamp_anchors.v1"
DEFAULT_TSR_URL = "https://freetsa.org/tsr"
CACERT_NAME = "freetsa_cacert.pem"
TSA_CERT_NAME = "freetsa_tsa.crt"
SHA256_ALG_OID = bytes.fromhex("0609608648016503040201")


def _der_len(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    b = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(b)]) + b


def _tlv(tag: int, body: bytes) -> bytes:
    return bytes([tag]) + _der_len(len(body)) + body


def _seq(body: bytes) -> bytes:
    return _tlv(0x30, body)


def build_tsq(file_sha256_hex: str) -> bytes:
    """Minimal DER ``TimeStampReq`` v1, sha256 imprint, certReq=TRUE."""
    digest = bytes.fromhex(file_sha256_hex)
    if len(digest) != 32:
        raise ValueError("sha256 hex must decode to 32 bytes")
    alg_id = _seq(SHA256_ALG_OID + _tlv(0x05, b""))  # OID + NULL
    imprint = _seq(alg_id + _tlv(0x04, digest))  # MessageImprint
    return _seq(_tlv(0x02, b"\x01") + imprint + _tlv(0x01, b"\xff"))


class DerError(ValueError):
    pass


def _read_tlv(buf: bytes, off: int) -> tuple[int, bytes, int]:
    """Return (tag, contents, next_offset); raises DerError on truncation."""
    if off >= len(buf):
        raise DerError("eof")
    tag = buf[off]
    off += 1
    if off >= len(buf):
        raise DerError("len eof")
    lb = buf[off]
    off += 1
    if lb & 0x80:
        n = lb & 0x7F
        if n == 0 or off + n > len(buf):
            raise DerError("long len")
        lb = int.from_bytes(buf[off : off + n], "big")
        off += n
    if off + lb > len(buf):
        raise DerError("body eof")
    return tag, buf[off : off + lb], off + lb


def _seq_children(body: bytes) -> list[tuple[int, bytes]]:
    out: list[tuple[int, bytes]] = []
    off = 0
    while off < len(body):
        tag, val, off = _read_tlv(body, off)
        out.append((tag, val))
    return out


def extract_imprint(tsr: bytes) -> bytes:
    """Pull the ``messageImprint`` digest out of a DER TimeStampResp.

    Path: TimeStampResp → status, [0] content → SignedData →
    encapContentInfo → [0] OCTET STRING(TSTInfo) → messageImprint →
    OCTET STRING digest.
    """
    try:
        tag, top, _ = _read_tlv(tsr, 0)
        if tag != 0x30:
            raise DerError("tsr not a sequence")
        children = _seq_children(top)
        if len(children) < 2:
            raise DerError("no token")
        tag, content = children[1]
        if tag != 0x30:
            raise DerError("contentInfo not a sequence")
        ci = _seq_children(content)
        # ci: OID, [0] signedData
        if len(ci) < 2 or ci[1][0] != 0xA0:
            raise DerError("signedData missing")
        _t, sd_wrap, _ = _read_tlv(ci[1][1], 0)
        if _t != 0x30:
            raise DerError("signedData not a sequence")
        sd = _seq_children(sd_wrap)
        # sd: version, digestAlgs SET, encapContentInfo SEQ, ...
        eci = None
        for t, v in sd:
            if t == 0x30:
                inner = _seq_children(v)
                if inner and inner[0][0] == 0x06:  # starts with OID
                    eci = inner
                    break
        if eci is None or len(eci) < 2 or eci[1][0] != 0xA0:
            raise DerError("encapContentInfo missing")
        _t, octs, _ = _read_tlv(eci[1][1], 0)
        if _t != 0x04:
            raise DerError("eContent not octet string")
        tst = _seq_children(_read_tlv(octs, 0)[1])
        # TSTInfo: version INT, policy OID, messageImprint SEQ
        for t, v in tst:
            if t == 0x30:
                mi = _seq_children(v)
                if len(mi) == 2 and mi[0][0] == 0x30 and mi[1][0] == 0x04 and len(mi[1][1]) == 32:
                    return mi[1][1]
        raise DerError("messageImprint not found")
    except (IndexError, DerError) as exc:
        raise DerError(f"malformed tsr: {exc}") from exc


def load_anchors(ts_dir: Path) -> dict[str, dict[str, str]]:
    path = ts_dir / ANCHORS_MANIFEST
    if not path.is_file():
        return {}
    try:
        body = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(body, dict) or body.get("schema") != ANCHORS_SCHEMA:
        return {}
    anchors = body.get("anchors", {})
    return anchors if isinstance(anchors, dict) else {}


def stamp_timestamp(
    target: str | Path,
    *,
    root: str | Path = ".",
    tsr_url: str = DEFAULT_TSR_URL,
    timeout: float = 30.0,
) -> Path:
    """Request a TSA token over ``target``'s sha256 and register the anchor.

    ``target`` is repo-relative; the token lands at
    ``quality/timestamps/<stem>.tsr`` and ``anchors.json`` records the
    digest it commits to (atomic writes both). Network call — only the
    32-byte digest leaves the machine.
    """
    import urllib.parse
    import urllib.request

    from quant_fund.utils.atomicio import atomic_write_bytes, atomic_write_text

    root_path = Path(root)
    target_rel = Path(target)
    digest = hash_bytes((root_path / target_rel).read_bytes())
    # Scheme allowlist before the request — urlopen would honor file:/ too.
    if urllib.parse.urlparse(tsr_url).scheme not in {"https", "http"}:
        raise ValueError(f"tsr_url scheme must be http(s): {tsr_url!r}")
    req = urllib.request.Request(
        tsr_url,
        data=build_tsq(digest),
        headers={"Content-Type": "application/timestamp-query"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  # nosec B310
        token = resp.read()
    # Fail closed: never register a token that doesn't commit to this digest.
    if extract_imprint(token) != bytes.fromhex(digest):
        raise ValueError("tsr token does not commit to the file digest")
    ts_dir = root_path / TIMESTAMPS_DIR
    ts_dir.mkdir(parents=True, exist_ok=True)
    name = target_rel.as_posix().replace("/", "__") + ".tsr"
    token_path = ts_dir / name
    atomic_write_bytes(token_path, token)
    anchors = load_anchors(ts_dir)
    anchors[name] = {"target": target_rel.as_posix(), "sha256": digest}
    atomic_write_text(
        ts_dir / ANCHORS_MANIFEST,
        json.dumps({"schema": ANCHORS_SCHEMA, "anchors": anchors}, indent=2, sort_keys=True) + "\n",
    )
    return token_path


def verify_timestamps(
    root: str | Path,
    *,
    ts_dir: Path = TIMESTAMPS_DIR,
) -> dict[str, Any]:
    """Verify committed timestamp anchors.

    ``anchored=False`` when no manifest exists (neutral state). Per anchor:
    token parses, token imprint equals the manifest's declared digest
    (``imprint_mismatch`` = forged token/manifest — hard error), target
    liveness (``anchor_target_missing``), ``fresh`` = token still commits to
    the target's *current* bytes (False = the pins moved on — the anchor
    remains valid proof of the earlier state). Chain verification via
    ``openssl ts -verify`` against the committed CA bundle when available.
    """
    root_path = Path(root)
    tdir = root_path / ts_dir
    manifest_path = tdir / ANCHORS_MANIFEST
    if not manifest_path.is_file():
        return {"ok": True, "anchored": False, "errors": []}
    anchors = load_anchors(tdir)
    if not anchors:
        return {
            "ok": False,
            "anchored": True,
            "errors": ["anchors_manifest_malformed"],
        }
    errors: list[str] = []
    fresh: dict[str, bool] = {}
    chain: dict[str, str] = {}
    cacert = tdir / CACERT_NAME
    tsa_cert = tdir / TSA_CERT_NAME
    openssl_ok = _openssl_available() and cacert.is_file()
    for name, entry in sorted(anchors.items()):
        token_path = tdir / name
        if not isinstance(entry, dict):
            errors.append(f"anchor_malformed:{name}")
            continue
        target_rel = str(entry.get("target", ""))
        declared = str(entry.get("sha256", ""))
        label = target_rel or name
        if not token_path.is_file():
            errors.append(f"tsr_missing:{name}")
            continue
        try:
            imprint = extract_imprint(token_path.read_bytes()).hex()
        except DerError:
            errors.append(f"tsr_malformed:{name}")
            continue
        if imprint != declared:
            errors.append(f"imprint_mismatch:{label}")
            continue
        target = root_path / target_rel
        resolved = target.resolve()
        if not resolved.is_relative_to(root_path.resolve()):
            # Manifest-declared path must stay inside the tree — otherwise
            # the freshness check hashes arbitrary host paths, a sealed
            # digest bit-oracle over files the corpus never covered.
            errors.append(f"anchor_target_uncontained:{label}")
            continue
        if not resolved.is_file():
            errors.append(f"anchor_target_missing:{label}")
            continue
        fresh[label] = hash_bytes(resolved.read_bytes()) == declared
        if openssl_ok:
            cmd = [
                "openssl",
                "ts",
                "-verify",
                "-digest",
                declared,
                "-in",
                str(token_path),
                "-CAfile",
                str(cacert),
            ]
            if tsa_cert.is_file():
                cmd += ["-untrusted", str(tsa_cert)]
            res = subprocess.run(cmd, capture_output=True, check=False)
            chain[label] = "chain_verified" if res.returncode == 0 else "chain_failed"
            if res.returncode != 0:
                errors.append(f"chain_verify_failed:{label}")
        else:
            chain[label] = "chain_unchecked"
    return {
        "ok": not errors,
        "anchored": True,
        "fresh": fresh,
        "chain": chain,
        "errors": sorted(errors),
    }


def _openssl_available() -> bool:
    try:
        res = subprocess.run(["openssl", "ts", "-help"], capture_output=True, check=False)
        return res.returncode in (0, 1)
    except OSError:
        return False
