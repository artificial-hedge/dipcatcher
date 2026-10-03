"""OpenTimestamps anchors — Bitcoin-anchored evidence via public calendars.

The RFC 3161 lane (``timestamp_anchor.py``) anchors pin bytes to a *trusted
timestamp authority's* signature time; a TSA is a single point of trust.
OpenTimestamps anchors the same digest to **Bitcoin**: a public calendar
aggregates submitted digests into a merkle tree whose root lands in a
coinbase ``OP_RETURN``. Once confirmed, rewriting the anchored state would
require rewriting Bitcoin — the strongest existence lower-bound available
without trusting any single party.

Layout under ``quality/timestamps/ots/``:

- ``<name>.ots`` — detached timestamp file (magic + sha256 digest + op chain).
- ``ots_anchors.json`` — ``ots_anchors.v1`` manifest mapping each ``.ots``
  token to ``{target, sha256}`` (frozen at stamp time).
- ``<name>.hdr`` — optional committed 80-byte block header. When present and
  a ``bitcoin:<height>`` attestation claims that height, the header's PoW is
  checked *purely* (sha256d < nBits target — no API trust at verify time).

Attestation states reported per anchor:

- ``pending:<calendar_uri>`` — digest accepted by a calendar, awaiting
  on-chain confirmation. Proves the digest reached the calendar; the Bitcoin
  claim is pending.
- ``bitcoin:<height>`` — the stamp claims inclusion in that block. Without a
  committed header this is ``inclusion_unverified`` (the calendar's claim);
  with one it reports ``pow_verified``/``pow_invalid``. ``pow_verified``
  means the claimed block header carries real proof-of-work (the block
  exists and was mined) — proving our digest is *inside* that block's
  coinbase additionally needs the block's transaction merkle path, which
  ``ots-upgrade`` does not fetch; the leaf→block binding stays the
  calendar's claim until a full-block check.

Honesty contract: no anchors = neutral; malformed tokens, imprint mismatches,
and trailing bytes are hard errors; a stale target reports ``fresh=False``
(valid proof of an earlier state, same as the TSA lane).
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import hash_bytes

OTS_DIR = Path("quality/timestamps/ots")
OTS_MANIFEST = Path("quality/timestamps/ots_anchors.json")
OTS_SCHEMA = "ots_anchors.v1"
OTS_MAGIC = b"\x00OpenTimestamps\x00\x01"
DEFAULT_CALENDARS = (
    "https://a.pool.opentimestamps.org",
    "https://b.pool.opentimestamps.org",
    "https://alice.btc.calendar.opentimestamps.org",
    "https://bob.btc.calendar.opentimestamps.org",
    "https://finney.calendar.eternitywall.com",
)

TAG_ATTESTATION = 0x00
TAG_FORK = 0xFF
OP_SHA256 = 0x08
OP_SHA1 = 0x02
OP_RIPEMD160 = 0x03
OP_KECCAK256 = 0x67
OP_APPEND = 0xF0
OP_PREPEND = 0xF1
OP_REVERSE = 0xF2
OP_HEXLIFY = 0xF3
ATT_PENDING = bytes.fromhex("83dfe30d2ef90c8e")
ATT_BITCOIN = bytes.fromhex("0588960d73d71901")

_CRYPT_OPS = {
    OP_SHA256: lambda m: hashlib.sha256(m).digest(),
    # SHA1/RIPEMD160 are legacy OTS proof ops, not security primitives here.
    OP_SHA1: lambda m: hashlib.sha1(m, usedforsecurity=False).digest(),
    OP_RIPEMD160: lambda m: hashlib.new("ripemd160", m).digest(),
}
_MAX_OP_ARG = 4096
_MAX_DEPTH = 128


class OtsError(ValueError):
    """Malformed or hostile .ots stream."""


def _read_varint(buf: bytes, off: int) -> tuple[int, int]:
    """LEB128 unsigned varint → (value, next_off)."""
    value = 0
    shift = 0
    while True:
        if off >= len(buf):
            raise OtsError("varint eof")
        b = buf[off]
        off += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, off
        shift += 7
        if shift > 35:
            raise OtsError("varint overflow")


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _read_varbytes(buf: bytes, off: int) -> tuple[bytes, int]:
    n, off = _read_varint(buf, off)
    if n > _MAX_OP_ARG or off + n > len(buf):
        raise OtsError("varbytes overrun")
    return buf[off : off + n], off + n


def _apply_op(tag: int, buf: bytes, off: int, msg: bytes) -> tuple[bytes, int]:
    """Apply one op to ``msg`` → (new_msg, next_off)."""
    if tag in _CRYPT_OPS:
        return _CRYPT_OPS[tag](msg), off
    if tag in (OP_APPEND, OP_PREPEND):
        arg, off = _read_varbytes(buf, off)
        return (msg + arg) if tag == OP_APPEND else (arg + msg), off
    if tag == OP_REVERSE:
        return msg[::-1], off
    if tag == OP_HEXLIFY:
        return msg.hex().encode(), off
    raise OtsError(f"unknown op tag 0x{tag:02x}")


def _parse_attestation(buf: bytes, off: int) -> tuple[dict[str, Any], int]:
    """``0x00`` + tag(8B) + varbytes(payload) → ({kind, ...}, next_off)."""
    if off + 8 > len(buf):
        raise OtsError("attestation tag eof")
    tag = buf[off : off + 8]
    off += 8
    payload, off = _read_varbytes(buf, off)
    if tag == ATT_PENDING:
        # Payload carries the calendar URI as an inner varbytes.
        uri, used = _read_varbytes(payload, 0)
        if used != len(payload):
            raise OtsError("pending attestation trailing payload")
        return {"kind": "pending", "uri": uri.decode("utf-8", "replace")}, off
    if tag == ATT_BITCOIN:
        # Payload is the claimed block height, a bare varint.
        height, used = _read_varint(payload, 0)
        if used != len(payload):
            raise OtsError("bitcoin attestation trailing payload")
        return {"kind": "bitcoin", "height": height}, off
    return {"kind": "unknown", "tag": tag.hex()}, off


def _parse_node(
    buf: bytes,
    off: int,
    msg: bytes,
    out: list[dict[str, Any]],
    depth: int,
) -> int:
    """Consume one timestamp node's items; record (digest, attestation) pairs.

    Wire format: ``[0xff item]*  item`` — 0xff prefixes every sibling except
    the node's final item. An op item recurses into its child node.
    """
    if depth > _MAX_DEPTH:
        raise OtsError("fork depth limit")
    while True:
        if off >= len(buf):
            raise OtsError("truncated stream")
        tag = buf[off]
        off += 1
        if tag == TAG_FORK:
            off = _parse_item(buf, off, msg, out, depth)
            continue
        return _parse_item_with_tag(buf, off, tag, msg, out, depth)


def _parse_item(buf: bytes, off: int, msg: bytes, out: list[dict[str, Any]], depth: int) -> int:
    if off >= len(buf):
        raise OtsError("fork item eof")
    return _parse_item_with_tag(buf, off + 1, buf[off], msg, out, depth)


def _parse_item_with_tag(
    buf: bytes, off: int, tag: int, msg: bytes, out: list[dict[str, Any]], depth: int
) -> int:
    return _parse_item_body(buf, off, tag, msg, out, depth)


def _parse_item_body(
    buf: bytes, off: int, tag: int, msg: bytes, out: list[dict[str, Any]], depth: int
) -> int:
    if tag == TAG_ATTESTATION:
        att, off = _parse_attestation(buf, off)
        att["committed_digest"] = msg.hex()
        out.append(att)
        return off
    child_msg, off = _apply_op(tag, buf, off, msg)
    return _parse_node(buf, off, child_msg, out, depth + 1)


def parse_ots(data: bytes, file_digest: bytes) -> list[dict[str, Any]]:
    """Parse a detached ``.ots`` file; replay ops over ``file_digest``.

    Returns the attestation list, each with the ``committed_digest`` it binds.
    The file's embedded hash must equal ``file_digest`` — a mismatch means the
    stamp was minted for different bytes (hard error upstream).
    """
    if len(data) < len(OTS_MAGIC) + 33 or not data.startswith(OTS_MAGIC):
        raise OtsError("bad magic/version")
    off = len(OTS_MAGIC)
    if data[off] != OP_SHA256:
        raise OtsError(f"unsupported file hash op 0x{data[off]:02x}")
    embedded = data[off + 1 : off + 33]
    if len(embedded) != 32:
        raise OtsError("truncated file digest")
    if embedded != file_digest:
        raise OtsError("file digest mismatch")
    off += 33
    out: list[dict[str, Any]] = []
    off = _parse_node(data, off, bytes(file_digest), out, 0)
    if off != len(data):
        raise OtsError("trailing bytes")
    if not out:
        raise OtsError("no attestations")
    return out


def _sha256d(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


def _bits_to_target(bits_le: bytes) -> int:
    bits = int.from_bytes(bits_le, "little")
    mantissa = bits & 0x007FFFFF  # drop sign bit
    exponent = bits >> 24
    return mantissa << (8 * (exponent - 3)) if exponent > 3 else mantissa >> (8 * (3 - exponent))


def verify_header_pow(header: bytes, claimed_height: int) -> dict[str, Any]:
    """Pure-Python proof-of-work check on an 80-byte Bitcoin block header.

    ``sha256d(header) < nBits target`` — self-validating, no chain access.
    Also extracts the header timestamp (block-time lower bound on the anchor).
    """
    if len(header) != 80:
        return {"ok": False, "error": "header_not_80_bytes"}
    bits = header[72:76]
    target = _bits_to_target(bits)
    h = int.from_bytes(_sha256d(header), "little")
    block_time = int.from_bytes(header[68:72], "little")
    ok = 0 < h <= target
    return {
        "ok": ok,
        "height": claimed_height,
        "block_time": block_time,
        "block_hash": _sha256d(header)[::-1].hex(),
        "error": None if ok else "pow_invalid",
    }


def stamp_ots(
    target: str | Path,
    *,
    root: str | Path = ".",
    calendars: tuple[str, ...] = DEFAULT_CALENDARS,
    timeout: float = 20.0,
) -> Path:
    """Submit ``target``'s sha256 to public OTS calendars; write the .ots file.

    Only the 32-byte digest leaves the machine. Responses are merged as
    ``0xff``-separated sibling branches (lexicographic order for
    determinism). Fails closed when no calendar answers and on any response
    that doesn't parse against the submitted digest.
    """
    from quant_fund.utils.atomicio import atomic_write_bytes, atomic_write_text

    root_path = Path(root)
    target_rel = Path(target)
    digest = hash_bytes((root_path / target_rel).read_bytes())
    digest_bytes = bytes.fromhex(digest)
    responses: list[bytes] = []
    for cal in calendars:
        url = cal.rstrip("/") + "/digest"
        try:
            req = urllib.request.Request(
                url,
                data=digest_bytes,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Accept": "application/vnd.opentimestamps.v1",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  # nosec B310
                body = resp.read()
            # The response is a timestamp stream rooted at our digest —
            # reject anything that doesn't parse or carries no attestation.
            probe: list[dict[str, Any]] = []
            _parse_node(body, 0, digest_bytes, probe, 0)
            if not probe:
                raise OtsError("calendar returned empty timestamp")
            responses.append(body)
        except OtsError:
            raise
        except (OSError, ValueError, KeyError):  # a dead calendar skips, not fails
            continue
    if not responses:
        raise ValueError("no OTS calendar answered")
    merged = (
        b"".join(b"\xff" + s for s in sorted(responses)[:-1]) + sorted(responses)[-1]
        if len(responses) > 1
        else responses[0]
    )
    detached = OTS_MAGIC + bytes([OP_SHA256]) + digest_bytes + merged
    ts_dir = root_path / OTS_DIR
    ts_dir.mkdir(parents=True, exist_ok=True)
    name = target_rel.as_posix().replace("/", "__") + ".ots"
    atomic_write_bytes(ts_dir / name, detached)
    manifest_path = root_path / OTS_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        manifest = {}
    anchors = manifest.get("anchors", {}) if isinstance(manifest, dict) else {}
    if not isinstance(anchors, dict):
        anchors = {}
    anchors[name] = {"target": target_rel.as_posix(), "sha256": digest}
    atomic_write_text(
        manifest_path,
        json.dumps({"schema": OTS_SCHEMA, "anchors": anchors}, indent=2, sort_keys=True) + "\n",
    )
    return ts_dir / name


DEFAULT_EXPLORER = "https://blockstream.info/api"
BLOCK_WITNESS_SUFFIX = ".blk"


def _read_varint_tx(tx: bytes, off: int) -> tuple[int, int]:
    """Bitcoin tx varint (not LEB128): 0xfd u16 / 0xfe u32 / 0xff u64."""
    if off >= len(tx):
        raise OtsError("tx varint eof")
    b = tx[off]
    off += 1
    if b < 0xFD:
        return b, off
    size = {0xFD: 2, 0xFE: 4, 0xFF: 8}[b]
    if off + size > len(tx):
        raise OtsError("tx varint eof")
    return int.from_bytes(tx[off : off + size], "little"), off + size


def _tx_output_scripts(tx: bytes) -> list[bytes]:
    """Extract every output's scriptPubKey from a raw transaction.

    Minimal Bitcoin tx parser: version(4) [+ segwit marker/flag] → vin →
    vout → [witness] → locktime(4). Only the script bytes matter to us."""
    if len(tx) < 10:
        raise OtsError("tx too short")
    off = 4
    segwit = tx[off] == 0x00 and tx[off + 1] == 0x01
    if segwit:
        off += 2
    vin_n, off = _read_varint_tx(tx, off)
    if vin_n == 0:
        raise OtsError("tx no inputs")
    for _ in range(vin_n):
        off += 36
        n, off = _read_varint_tx(tx, off)
        off += n + 4
    vout_n, off = _read_varint_tx(tx, off)
    scripts: list[bytes] = []
    for _ in range(vout_n):
        off += 8
        n, off = _read_varint_tx(tx, off)
        scripts.append(tx[off : off + n])
        off += n
    if off > len(tx):
        raise OtsError("tx truncated")
    return scripts


def _script_pushes(script: bytes) -> list[bytes]:
    """Data pushes on a scriptPubKey — the OP_RETURN payload carriers."""
    out: list[bytes] = []
    off = 0
    while off < len(script):
        op = script[off]
        off += 1
        if op == 0x00:
            continue
        if op <= 0x4B:
            n = op
        elif op == 0x4C:
            n, off = script[off], off + 1
        elif op == 0x4D:
            n, off = int.from_bytes(script[off : off + 2], "little"), off + 2
        else:
            break  # non-push opcode — nothing else we care about
        if off + n > len(script):
            break
        out.append(script[off : off + n])
        off += n
    return out


def _tx_base_serialization(tx: bytes) -> bytes:
    """Strip a segwit serialization down to the txid-hashed form.

    A txid is sha256d over ``version || vin || vout || locktime`` — the
    marker/flag and per-input witness stacks are excluded. Explorers serve
    the witness-inclusive form, so hashing raw bytes yields the *wtxid*,
    not the txid the block commits to."""
    if len(tx) < 10:
        raise OtsError("tx too short")
    if not (tx[4] == 0x00 and tx[5] == 0x01):
        return tx
    off = 6
    vin_n, off = _read_varint_tx(tx, off)
    if vin_n == 0:
        raise OtsError("tx no inputs")
    for _ in range(vin_n):
        off += 36
        n, off = _read_varint_tx(tx, off)
        off += n + 4
    vout_n, off = _read_varint_tx(tx, off)
    for _ in range(vout_n):
        off += 8
        n, off = _read_varint_tx(tx, off)
        off += n
    if off > len(tx) - 4:
        raise OtsError("tx truncated")
    return tx[:4] + tx[6:off] + tx[-4:]


def coinbase_height(coinbase_raw: bytes) -> int | None:
    """BIP34 block height — the first scriptsig push of a coinbase input.

    Every post-227836 coinbase declares its own height, binding the claimed
    height to the block contents rather than the filename or manifest."""
    try:
        off = 4
        if coinbase_raw[off] == 0x00 and coinbase_raw[off + 1] == 0x01:
            off += 2
        vin_n, off = _read_varint_tx(coinbase_raw, off)
        if vin_n == 0:
            return None
        off += 36  # prevhash + vout of vin[0]
        n, off = _read_varint_tx(coinbase_raw, off)
        scriptsig = coinbase_raw[off : off + n]
        if not scriptsig:
            return None
        push_n = scriptsig[0]
        if push_n < 1 or push_n > 5 or len(scriptsig) < 1 + push_n:
            return None
        return int.from_bytes(scriptsig[1 : 1 + push_n], "little")
    except (OtsError, IndexError):
        return None


def coinbase_commitments(coinbase_raw: bytes) -> set[bytes]:
    """OP_RETURN payloads embedded in a coinbase transaction.

    An OTS calendar writes its aggregated merkle root as an OP_RETURN push —
    this is the leaf→block binding: the .ots attestation's committed_digest
    must appear here verbatim."""
    commitments: set[bytes] = set()
    for script in _tx_output_scripts(coinbase_raw):
        if script and script[0] == 0x6A:  # OP_RETURN
            commitments.update(_script_pushes(script[1:]))
    return commitments


def block_merkle_root(txids_internal: list[bytes]) -> bytes:
    """Bitcoin block merkle root over internal-order txids (dup-last rule)."""
    if not txids_internal:
        raise OtsError("empty txid list")
    level = txids_internal
    while len(level) > 1:
        if len(level) & 1:
            level = [*level, level[-1]]
        level = [_sha256d(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def verify_block_inclusion(
    header: bytes,
    txids: list[str],
    coinbase_raw: bytes,
    committed_digest: bytes,
    *,
    claimed_height: int | None = None,
) -> dict[str, Any]:
    """Full leaf→block check — zero trust in the calendar's claim.

    1. sha256d(witness-stripped coinbase) is txid[0] (coinbase position by
       Bitcoin rule).
    2. Merkle root over all txids equals header[36:68] — the tx set is proven
       by the header's own commitment.
    3. The coinbase carries an OP_RETURN push equal to ``committed_digest``
       (the value the OTS attestation binds) — our aggregated leaf is inside
       the coinbase, hence inside this block.
    4. When ``claimed_height`` is given, the coinbase's BIP34 push must equal
       it — a valid witness for a *different* block can't be relabeled.
    """
    if len(header) != 80:
        return {"ok": False, "error": "header_not_80_bytes"}
    try:
        internals = [bytes.fromhex(t)[::-1] for t in txids]
        coinbase_txid = _sha256d(_tx_base_serialization(coinbase_raw))
        if not internals or internals[0] != coinbase_txid:
            return {"ok": False, "error": "coinbase_txid_mismatch"}
        if block_merkle_root(internals) != header[36:68]:
            return {"ok": False, "error": "block_merkle_mismatch"}
        if committed_digest not in coinbase_commitments(coinbase_raw):
            return {"ok": False, "error": "commitment_absent"}
        if claimed_height is not None:
            h = coinbase_height(coinbase_raw)
            if h is None or h != claimed_height:
                return {"ok": False, "error": "coinbase_height_mismatch"}
    except (ValueError, OtsError) as exc:
        return {"ok": False, "error": f"block_parse:{exc}"}
    return {"ok": True, "n_tx": len(txids), "coinbase_txid": coinbase_txid[::-1].hex()}


def verify_burial(header: bytes, successors: list[bytes]) -> dict[str, Any]:
    """SPV burial check: K committed successors each link prev-hash and pass
    PoW — the anchored block is buried under K blocks' cumulative work.

    Without this, a fabricated valid-PoW header from any historical period
    could pair with a self-consistent .blk witness; linkage to a live chain
    tip is what pins real elapsed work. Returns cumulative work bits —
    each block contributes 2^256/target expected hashes → log2 of the sum.
    """
    import math

    chain = [header, *successors]
    for i, hdr in enumerate(chain):
        if len(hdr) != 80:
            return {"ok": False, "error": f"succ_header_not_80_bytes:{i}"}
        if not verify_header_pow(hdr, 0)["ok"]:
            return {"ok": False, "error": f"succ_pow_invalid:{i}"}
        if i and hdr[4:36] != _sha256d(chain[i - 1]):
            return {"ok": False, "error": f"chain_break:{i}"}
    work = 0.0
    for hdr in chain:
        target = _bits_to_target(hdr[72:76])
        work += (2.0**256 / (target + 1)) if target else 0.0
    return {
        "ok": True,
        "depth": len(successors),
        "work_log2": round(math.log2(work), 1) if work else 0.0,
    }


def _get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  # nosec B310
        return bytes(resp.read())


def _fetch_header(height: int, *, explorer: str, timeout: float) -> bytes | None:
    """block-height → block hash → 80-byte header, via the explorer API."""
    try:
        block_hash = _get(explorer.rstrip("/") + f"/block-height/{height}", timeout).strip()
        raw = _get(explorer.rstrip("/") + f"/block/{block_hash.decode()}/header", timeout).strip()
        header = bytes.fromhex(raw.decode())
    except Exception:  # noqa: BLE001 — explorer outage skips, not fails
        return None
    return header if len(header) == 80 else None


def _fetch_block_witness(
    height: int, *, explorer: str, timeout: float, bury: int = 0
) -> dict[str, Any] | None:
    """txid list + raw coinbase (+ K successor headers) — the full witness."""
    try:
        base = explorer.rstrip("/")
        block_hash = _get(f"{base}/block-height/{height}", timeout).strip().decode()
        txids = json.loads(_get(f"{base}/block/{block_hash}/txids", timeout))
        if (
            not isinstance(txids, list)
            or not txids
            or not all(isinstance(t, str) and len(t) == 64 for t in txids)
        ):
            return None
        coinbase = _get(f"{base}/tx/{txids[0]}/raw", timeout)
        out: dict[str, Any] = {"txids": txids, "coinbase": coinbase.hex()}
        if bury > 0:
            succ: list[str] = []
            for h in range(height + 1, height + bury + 1):
                sh = _get(f"{base}/block-height/{h}", timeout).strip().decode()
                raw = _get(f"{base}/block/{sh}/header", timeout).strip().decode()
                succ_hdr = bytes.fromhex(raw)
                if len(succ_hdr) != 80:
                    return None
                succ.append(succ_hdr.hex())
            out["succ_headers"] = succ
        return out
    except Exception:  # noqa: BLE001 — explorer outage skips, not fails
        return None


def upgrade_ots(
    *,
    root: str | Path = ".",
    ots_dir: Path = OTS_DIR,
    manifest: Path = OTS_MANIFEST,
    explorer: str = DEFAULT_EXPLORER,
    timeout: float = 20.0,
    full: bool = False,
    bury: int = 0,
) -> dict[str, Any]:
    """Upgrade pending anchors: poll calendars for Bitcoin confirmations.

    For each manifest entry whose proof is still ``pending``, query every
    calendar URI named in its own attestations for the upgraded timestamp
    (``GET <calendar>/timestamp/<digest-hex>``). A response that parses
    against the anchored digest and carries a ``bitcoin`` attestation
    replaces the ``.ots`` (atomic write); the claimed block header is then
    fetched from ``explorer`` and committed as ``<name>.<height>.hdr`` so
    ``verify_ots`` can self-verify its PoW — no API trust at verify time.

    States per anchor: ``upgraded:<height>``, ``still_pending``,
    ``no_bitcoin_attestation``, ``upgrade_malformed:<e>``, ``hdr_unavailable``.
    """
    from quant_fund.utils.atomicio import atomic_write_bytes

    root_path = Path(root)
    tdir = root_path / ots_dir
    try:
        body = json.loads((root_path / manifest).read_text())
    except (OSError, ValueError):
        body = {}
    anchors = body.get("anchors") if body.get("schema") == OTS_SCHEMA else None
    if not isinstance(anchors, dict) or not anchors:
        return {"ok": False, "upgraded": [], "errors": ["ots_manifest_malformed"]}
    results: dict[str, list[str]] = {}
    upgraded: list[str] = []
    errors: list[str] = []
    for name, entry in sorted(anchors.items()):
        label = str(entry.get("target", "")) or name
        token_path = tdir / name
        declared = str(entry.get("sha256", ""))
        states: list[str] = []
        try:
            digest_bytes = bytes.fromhex(declared)
            atts = parse_ots(token_path.read_bytes(), digest_bytes)
        except (OtsError, ValueError, OSError) as exc:
            errors.append(f"ots_malformed:{name}:{exc}")
            results[label] = [f"ots_malformed:{exc}"]
            continue
        calendars = sorted({a["uri"] for a in atts if a["kind"] == "pending"})
        if any(a["kind"] == "bitcoin" for a in atts):
            states.append("already_bitcoin")
        elif not calendars:
            states.append("still_pending")
        else:
            for cal in calendars:
                try:
                    body2 = _get(cal.rstrip("/") + f"/timestamp/{declared}", timeout)
                    probe: list[dict[str, Any]] = []
                    off = _parse_node(body2, 0, digest_bytes, probe, 0)
                    if off != len(body2):
                        raise OtsError("trailing bytes")
                except OtsError as exc:
                    states.append(f"upgrade_malformed:{exc}")
                    continue
                except Exception:  # noqa: BLE001 — dead calendar skips
                    continue
                heights = [a["height"] for a in probe if a["kind"] == "bitcoin"]
                if not heights:
                    states.append("no_bitcoin_attestation")
                    continue
                detached = OTS_MAGIC + bytes([OP_SHA256]) + digest_bytes + body2
                atomic_write_bytes(token_path, detached)
                for height in sorted(set(heights)):
                    hdr = _fetch_header(int(height), explorer=explorer, timeout=timeout)
                    if hdr is None:
                        states.append(f"bitcoin:{height}:hdr_unavailable")
                    elif full:
                        bw = _fetch_block_witness(
                            int(height), explorer=explorer, timeout=timeout, bury=bury
                        )
                        if bw is None:
                            states.append(f"bitcoin:{height}:block_witness_unavailable")
                        else:
                            atomic_write_bytes(
                                tdir / f"{Path(name).stem}.{height}{BLOCK_WITNESS_SUFFIX}",
                                json.dumps(bw).encode(),
                            )
                            atomic_write_bytes(tdir / f"{Path(name).stem}.{height}.hdr", hdr)
                            states.append(f"upgraded:{height}:full")
                    else:
                        atomic_write_bytes(tdir / f"{Path(name).stem}.{height}.hdr", hdr)
                        states.append(f"upgraded:{height}")
                upgraded.append(name)
                break
            else:
                if not states:
                    states.append("still_pending")
        results[label] = states
    return {"ok": not errors, "upgraded": upgraded, "anchors": results, "errors": sorted(errors)}


def verify_ots(
    root: str | Path,
    *,
    ots_dir: Path = OTS_DIR,
    manifest: Path = OTS_MANIFEST,
) -> dict[str, Any]:
    """Verify committed OTS anchors. Mirrors ``verify_timestamps`` semantics."""
    root_path = Path(root)
    tdir = root_path / ots_dir
    manifest_path = root_path / manifest
    if not manifest_path.is_file():
        return {"ok": True, "anchored": False, "errors": []}
    try:
        body = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        body = {}
    anchors = body.get("anchors") if body.get("schema") == OTS_SCHEMA else None
    if not isinstance(anchors, dict) or not anchors:
        return {"ok": False, "anchored": True, "errors": ["ots_manifest_malformed"]}
    errors: list[str] = []
    fresh: dict[str, bool] = {}
    attestations: dict[str, list[str]] = {}
    for name, entry in sorted(anchors.items()):
        label = str(entry.get("target", "")) or name
        token_path = tdir / name
        if not token_path.is_file():
            errors.append(f"ots_missing:{name}")
            continue
        declared = str(entry.get("sha256", ""))
        try:
            atts = parse_ots(token_path.read_bytes(), bytes.fromhex(declared))
        except (OtsError, ValueError) as exc:
            errors.append(f"ots_malformed:{name}:{exc}")
            continue
        target = root_path / str(entry.get("target", ""))
        if not target.is_file():
            errors.append(f"anchor_target_missing:{label}")
            continue
        fresh[label] = hash_bytes(target.read_bytes()) == declared
        states: list[str] = []
        for att in atts:
            if att["kind"] == "pending":
                states.append(f"pending:{att['uri']}")
            elif att["kind"] == "bitcoin":
                hdr = tdir / f"{Path(name).stem}.{att['height']}.hdr"
                blk = tdir / f"{Path(name).stem}.{att['height']}{BLOCK_WITNESS_SUFFIX}"
                if hdr.is_file():
                    pow_res = verify_header_pow(hdr.read_bytes(), int(att["height"]))
                    if not pow_res["ok"]:
                        states.append(f"bitcoin:{att['height']}:pow_invalid")
                        errors.append(f"pow_invalid:{label}")
                        continue
                    if blk.is_file():
                        try:
                            bw = json.loads(blk.read_text())
                            inc = verify_block_inclusion(
                                hdr.read_bytes(),
                                bw["txids"],
                                bytes.fromhex(bw["coinbase"]),
                                bytes.fromhex(att["committed_digest"]),
                                claimed_height=int(att["height"]),
                            )
                        except (OSError, ValueError, KeyError) as exc:
                            inc = {"ok": False, "error": f"blk_parse:{exc.__class__.__name__}"}
                        if inc["ok"]:
                            states.append(f"bitcoin:{att['height']}:fully_verified")
                            succ = bw.get("succ_headers")
                            if isinstance(succ, list) and succ:
                                burial = verify_burial(
                                    hdr.read_bytes(),
                                    [bytes.fromhex(s) for s in succ],
                                )
                                if burial["ok"]:
                                    states.append(
                                        f"buried:{burial['depth']}:work_log2={burial['work_log2']}"
                                    )
                                else:
                                    errors.append(f"burial_invalid:{label}:{burial['error']}")
                        else:
                            states.append(
                                f"bitcoin:{att['height']}:inclusion_invalid:{inc['error']}"
                            )
                            errors.append(f"inclusion_invalid:{label}")
                    else:
                        states.append(f"bitcoin:{att['height']}:pow_verified")
                else:
                    states.append(f"bitcoin:{att['height']}:inclusion_unverified")
            else:
                states.append(f"unknown:{att.get('tag', '')}")
        attestations[label] = states
    return {
        "ok": not errors,
        "anchored": True,
        "fresh": fresh,
        "attestations": attestations,
        "errors": sorted(errors),
    }
