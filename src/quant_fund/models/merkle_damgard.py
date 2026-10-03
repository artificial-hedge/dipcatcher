"""Merkle–Damgård iterated hash + length-extension demonstration (toy 64-bit)."""

import numpy as np

_SEED = 20261231 + 563

_MASK = (1 << 64) - 1


def _f(h: int, m: int) -> int:
    # compression toy: h' = (h*K + m*K2 + h^m) rotl13 mod 2^64
    x = (h * 0x9E3779B97F4A7C15 + m * 0xC2B2AE3D27D4EB4F + (h ^ m)) & _MASK
    return ((x << 13) | (x >> 51)) & _MASK


def md_hash(data: bytes, iv: int = 0xCBF29CE484222325) -> int:
    """MD with 8-byte blocks + 8-byte length padding."""
    b = bytearray(data)
    b += len(data).to_bytes(8, "big")  # length padding
    while len(b) % 8:
        b.append(0)
    h = iv
    for i in range(0, len(b), 8):
        h = _f(h, int.from_bytes(b[i : i + 8], "big"))
    return h


def _glue_pad_len(n: int) -> int:
    total = n + 8
    return (8 - total % 8) % 8


def extend(digest: int, suffix: bytes, orig_len: int) -> int:
    """Length extension: continue hashing from digest as state."""
    pad = _glue_pad_len(orig_len)
    forged_prefix_len = orig_len + 8 + pad + len(suffix)
    b = bytearray(suffix)
    b += forged_prefix_len.to_bytes(8, "big")
    while len(b) % 8:
        b.append(0)
    h = digest
    for i in range(0, len(b), 8):
        h = _f(h, int.from_bytes(b[i : i + 8], "big"))
    return h


def bench_merkle_damgard(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = 40
    det = ext_ok = 0
    for _ in range(n):
        m = bytes(rng.randint(0, 256) for _ in range(rng.randint(1, 24)))
        det += int(md_hash(m) == md_hash(m))
        d = md_hash(m)
        suffix = b"|x"
        # attacker forges hash(m || gluepad || suffix)
        glue = _glue_pad_len(len(m))
        forged = extend(d, suffix, len(m))
        real = md_hash(m + len(m).to_bytes(8, "big") + b"\x00" * glue + suffix)
        ext_ok += int(forged == real)
    return {
        "synthetic_deterministic": float(det / n),
        "synthetic_length_ext_demo": float(ext_ok / n),
    }
