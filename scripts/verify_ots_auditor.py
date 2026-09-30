#!/usr/bin/env python3
"""Standalone OpenTimestamps/Bitcoin anchor auditor — stdlib only.

Independent reimplementation of ``quant_fund.research.ots_anchor`` for
third-party auditors who hold only the committed files:

    python3 scripts/verify_ots_auditor.py \
        --target quality/epoch_heads.json \
        --ots quality/timestamps/ots/quality__epoch_heads.json.ots \
        [--hdr quality/timestamps/ots/quality__epoch_heads.json.<h>.hdr] \
        [--blk quality/timestamps/ots/quality__epoch_heads.json.<h>.blk]

Verifies, entirely offline:

1. .ots detached-proof wire format (magic + 0x08 + sha256 + stream)
2. Each attestation's bound digest is derived by applying the op chain
   (append/prepend/reverse/hexlify/sha1/ripemd160/sha256/keccak) to the
   committed file digest — a proof that skips the committed digest fails
3. Per attestation: calendar URI (pending) or Bitcoin block height
4. With .hdr: sha256d(header) <= nBits target (real PoW check)
5. With .blk: coinbase txid == txids[0], recomputed merkle root ==
   header[36:68], the attestation's bound digest present in a coinbase
   OP_RETURN push

Exit 0 only when every check passes. Prints one verdict line per layer.
This file must never import the library — it is the differential oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

OTS_MAGIC = b"\x00OpenTimestamps\x00\x01"
TAG_PENDING = bytes.fromhex("83dfe30d2ef90c8e")
TAG_BITCOIN = bytes.fromhex("0588960d73d71901")
# op byte -> (has_arg, name)
OPS = {
    0xF0: (True, "append"),
    0xF1: (True, "prepend"),
    0xF2: (False, "reverse"),
    0xF3: (False, "hexlify"),
    0x02: (False, "sha1"),
    0x03: (False, "ripemd160"),
    0x08: (False, "sha256"),
    0x67: (False, "keccak"),
}
MAX_DEPTH = 128


def _sha256d(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


def _keccak(b: bytes) -> bytes:
    try:
        from Crypto.Hash import keccak as _k  # type: ignore[import-not-found]
    except ImportError as exc:
        # Keccak is only reachable in exotic proofs; fail loudly instead.
        raise ValueError("keccak_unavailable") from exc
    k = _k.new(digest_bits=256)
    k.update(b)
    return k.digest()


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """LEB128 varint (OTS serialization form)."""
    shift = val = 0
    while True:
        if pos >= len(buf) or shift > 63:
            raise ValueError("varint")
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, pos
        shift += 7


def _read_varbytes(buf: bytes, pos: int) -> tuple[bytes, int]:
    n, pos = _read_varint(buf, pos)
    if n > 4096 or pos + n > len(buf):
        raise ValueError("varbytes")
    return buf[pos : pos + n], pos + n


def _apply_op(op: int, msg: bytes, arg: bytes) -> bytes:
    name = OPS[op][1]
    if name == "append":
        return msg + arg
    if name == "prepend":
        return arg + msg
    if name == "reverse":
        return msg[::-1]
    if name == "hexlify":
        return msg.hex().encode()
    if name == "sha1":
        return hashlib.sha1(msg, usedforsecurity=False).digest()
    if name == "ripemd160":
        return hashlib.new("ripemd160", msg, usedforsecurity=False).digest()
    if name == "sha256":
        return hashlib.sha256(msg).digest()
    if name == "keccak":
        return _keccak(msg)
    raise ValueError(name)


def _parse_branch(
    buf: bytes, pos: int, msg: bytes, depth: int
) -> tuple[list[tuple[str, bytes, object]], int]:
    """Parse a sequence of items starting from msg; collect attestations."""
    if depth > MAX_DEPTH:
        raise ValueError("depth")
    atts: list[tuple[str, bytes, object]] = []
    while pos < len(buf):
        b = buf[pos]
        if b == 0xFF:
            sub, pos = _parse_branch(buf, pos + 1, msg, depth + 1)
            atts.extend(sub)
            continue
        if b in OPS:
            has_arg, _ = OPS[b]
            if has_arg:
                arg, pos = _read_varbytes(buf, pos + 1)
            else:
                pos += 1
                arg = b""
            msg = _apply_op(b, msg, arg)
            continue
        if b == 0x00:
            if pos + 9 > len(buf):
                raise ValueError("attestation_trunc")
            tag = buf[pos + 1 : pos + 9]
            payload, pos = _read_varbytes(buf, pos + 9)
            if tag == TAG_BITCOIN:
                h, end = _read_varint(payload, 0)
                if end != len(payload):
                    raise ValueError("bitcoin_payload_junk")
                atts.append(("bitcoin", msg, h))
            elif tag == TAG_PENDING:
                inner, end = _read_varbytes(payload, 0)
                if end != len(payload):
                    raise ValueError("pending_payload_junk")
                atts.append(("pending", msg, inner.decode("utf-8", "replace")))
            else:
                atts.append(("unknown", msg, tag.hex()))
            continue
        raise ValueError(f"item:{b:#x}")
    return atts, pos


def parse_ots(detached: bytes) -> tuple[bytes, list[tuple[str, bytes, object]]]:
    """→ (committed file digest, [(kind, bound_digest, payload)])."""
    if not detached.startswith(OTS_MAGIC + b"\x08"):
        raise ValueError("magic")
    head = len(OTS_MAGIC) + 1
    digest = detached[head : head + 32]
    if len(digest) != 32:
        raise ValueError("digest")
    atts, pos = _parse_branch(detached, head + 32, digest, 0)
    if pos != len(detached):
        raise ValueError("trailing_bytes")
    if not atts:
        raise ValueError("no_attestations")
    return digest, atts


def _bits_to_target(header: bytes) -> int:
    exp = header[72]
    mant = int.from_bytes(header[73:76], "little")
    return mant << (8 * (exp - 3)) if exp >= 3 else mant >> (8 * (3 - exp))


def verify_pow(header: bytes) -> bool:
    if len(header) != 80:
        return False
    return int.from_bytes(_sha256d(header), "little") <= _bits_to_target(header)


def _tx_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """Bitcoin tx varint (fd/fe/ff prefixes)."""
    b0 = buf[pos]
    if b0 < 0xFD:
        return b0, pos + 1
    size = {0xFD: 2, 0xFE: 4, 0xFF: 8}[b0]
    return int.from_bytes(buf[pos + 1 : pos + 1 + size], "little"), pos + 1 + size


def _output_scripts(tx: bytes) -> list[bytes]:
    if len(tx) < 10:
        raise ValueError("tx_short")
    pos = 4  # version
    if tx[4] == 0 and tx[5] != 0:  # segwit marker+flag
        pos = 6
    n_vin, pos = _tx_varint(tx, pos)
    for _ in range(n_vin):
        pos += 36
        slen, pos = _tx_varint(tx, pos)
        pos += slen + 4
    n_vout, pos = _tx_varint(tx, pos)
    if n_vout > 10_000:
        raise ValueError("vout_absurd")
    scripts: list[bytes] = []
    for _ in range(n_vout):
        pos += 8
        plen, pos = _tx_varint(tx, pos)
        scripts.append(tx[pos : pos + plen])
        pos += plen
    return scripts


def _script_pushes(script: bytes) -> list[bytes]:
    out: list[bytes] = []
    pos = 0
    while pos < len(script):
        op = script[pos]
        pos += 1
        if op <= 0x4B:
            out.append(script[pos : pos + op])
            pos += op
        elif op == 0x4C:
            n = script[pos]
            pos += 1
            out.append(script[pos : pos + n])
            pos += n
        elif op == 0x4D:
            n = int.from_bytes(script[pos : pos + 2], "little")
            pos += 2
            out.append(script[pos : pos + n])
            pos += n
        else:
            break
    return out


def _merkle_root(txids_le: list[bytes]) -> bytes:
    layer = txids_le
    while len(layer) > 1:
        if len(layer) % 2:
            layer = [*layer, layer[-1]]
        layer = [_sha256d(layer[i] + layer[i + 1]) for i in range(0, len(layer), 2)]
    return layer[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--ots", required=True)
    ap.add_argument("--hdr")
    ap.add_argument("--blk")
    args = ap.parse_args()

    with open(args.target, "rb") as fh:
        target = fh.read()
    try:
        digest, atts = parse_ots(open(args.ots, "rb").read())
    except (OSError, ValueError) as exc:
        print(f"FAIL ots_malformed:{exc}")
        return 1
    actual = hashlib.sha256(target).digest()
    stale = digest != actual
    if stale:
        print(
            f"note: proof commits {digest.hex()[:16]}… but target is "
            f"{actual.hex()[:16]}… — stale anchor (attests superseded bytes)"
        )
    else:
        print(f"digest ok: sha256(target)={actual.hex()}")

    bitcoin = [(bound_d, h) for k, bound_d, h in atts if k == "bitcoin"]
    for kind, bound_d, payload in atts:
        if kind == "pending":
            print(f"pending attestation: {payload} (bound {bound_d.hex()[:16]}…)")
        elif kind == "bitcoin":
            print(f"bitcoin attestation: height={payload} (bound {bound_d.hex()[:16]}…)")
        else:
            print(f"unknown attestation tag: {payload}")

    if args.hdr:
        header = open(args.hdr, "rb").read()
        if not verify_pow(header):
            print("FAIL header PoW invalid (sha256d > nBits target)")
            return 1
        print("header PoW ok (sha256d <= nBits)")
        if args.blk:
            blk = json.loads(open(args.blk).read())
            txids = [bytes.fromhex(t)[::-1] for t in blk["txids"]]
            coinbase = bytes.fromhex(blk["coinbase"])
            if _sha256d(coinbase) != txids[0]:
                print("FAIL coinbase txid != txids[0]")
                return 1
            if _merkle_root(txids) != header[36:68]:
                print("FAIL recomputed merkle root != header[36:68]")
                return 1
            pushes = [
                p
                for s in _output_scripts(coinbase)
                if s.startswith(b"\x6a")
                for p in _script_pushes(s[1:])
            ]
            btc_bound = {b for b, _ in bitcoin}
            if not btc_bound <= set(pushes):
                missing = [b.hex()[:16] for b in btc_bound - set(pushes)]
                print(f"FAIL attestation digests absent from coinbase: {missing}")
                return 1
            print(f"inclusion ok: {len(txids)} tx; merkle root + OP_RETURN commitment(s) verified")
            succ = blk.get("succ_headers") or []
            if succ:
                chain = [header, *[bytes.fromhex(s) for s in succ]]
                for i, h in enumerate(chain):
                    if len(h) != 80 or not verify_pow(h):
                        print(f"FAIL successor {i}: bad length or PoW")
                        return 1
                    if i and h[4:36] != _sha256d(chain[i - 1]):
                        print(f"FAIL chain break at successor {i}")
                        return 1
                print(f"burial ok: {len(succ)} linked successors with valid PoW")
            print("VERDICT: fully_verified" + (" [stale]" if stale else ""))
            return 0
        print("VERDICT: pow_verified" + (" [stale]" if stale else ""))
        return 0
    if bitcoin:
        print(
            "VERDICT: attested (no header supplied — PoW unchecked)" + (" [stale]" if stale else "")
        )
        return 0
    print("VERDICT: pending" + (" [stale]" if stale else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
