"""Encrypt-then-MAC AEAD toy: CTR-XOR keystream + HMAC over (nonce, ct, ad)."""

import hashlib
import hmac as _hmac

import numpy as np

_SEED = 20261231 + 562


def _keystream(key: bytes, nonce: bytes, n: int) -> bytes:
    out = b""
    ctr = 0
    while len(out) < n:
        out += hashlib.sha256(key + nonce + ctr.to_bytes(4, "big")).digest()
        ctr += 1
    return out[:n]


def seal(k_enc: bytes, k_mac: bytes, nonce: bytes, pt: bytes, ad: bytes) -> bytes:
    ct = bytes(a ^ b for a, b in zip(pt, _keystream(k_enc, nonce, len(pt)), strict=True))
    tag = _hmac.new(k_mac, nonce + ad + ct, hashlib.sha256).digest()[:16]
    return nonce + ct + tag


def open_(k_enc: bytes, k_mac: bytes, blob: bytes, ad: bytes) -> bytes | None:
    if len(blob) < 8 + 16:
        return None
    nonce, ct, tag = blob[:8], blob[8:-16], blob[-16:]
    exp = _hmac.new(k_mac, nonce + ad + ct, hashlib.sha256).digest()[:16]
    if not _hmac.compare_digest(tag, exp):
        return None
    return bytes(a ^ b for a, b in zip(ct, _keystream(k_enc, nonce, len(ct)), strict=True))


def bench_aead_etm(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    k_enc = bytes(rng.randint(0, 256) for _ in range(32))
    k_mac = bytes(rng.randint(0, 256) for _ in range(32))
    n = 60
    rt = tam = ad_fail = 0
    for _ in range(n):
        pt = bytes(rng.randint(0, 256) for _ in range(rng.randint(0, 64)))
        ad = bytes(rng.randint(0, 256) for _ in range(8))
        nonce = bytes(rng.randint(0, 256) for _ in range(8))
        blob = seal(k_enc, k_mac, nonce, pt, ad)
        rt += int(open_(k_enc, k_mac, blob, ad) == pt)
        i = rng.randint(0, len(blob))
        bad = blob[:i] + bytes([blob[i] ^ 1]) + blob[i + 1 :]
        tam += int(open_(k_enc, k_mac, bad, ad) is None)
        ad2 = bytes([ad[0] ^ 1]) + ad[1:]
        ad_fail += int(open_(k_enc, k_mac, blob, ad2) is None)
    return {
        "synthetic_roundtrip": float(rt / n),
        "synthetic_tamper_reject": float(tam / n),
        "synthetic_ad_binding": float(ad_fail / n),
    }
