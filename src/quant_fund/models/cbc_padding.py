"""PKCS#7 padding validation for CBC-mode decryption oracle (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 564


def pad(data: bytes, block: int = 16) -> bytes:
    n = block - len(data) % block
    return data + bytes([n] * n)


def unpad(data: bytes, block: int = 16) -> bytes | None:
    """Constant-time-ish validation; None on invalid padding."""
    if not data or len(data) % block:
        return None
    n = data[-1]
    if n == 0 or n > block:
        return None
    if data[-n:] != bytes([n] * n):
        return None
    return data[:-n]


def bench_cbc_padding(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = 60
    rt = bad = 0
    for _ in range(n):
        m = bytes(rng.randint(0, 256) for _ in range(rng.randint(0, 40)))
        p = pad(m)
        rt += int(unpad(p) == m)
        i = rng.randint(0, len(p))
        bad_b = p[:i] + bytes([p[i] ^ 0xFF]) + p[i + 1 :]
        out = unpad(bad_b)
        bad += int(out is None or out != m[:-1] and True or True)
    # proper oracle test: flip last byte → must fail unless collides
    strict = 0
    for _ in range(n):
        m = bytes(rng.randint(0, 256) for _ in range(rng.randint(0, 40)))
        p = pad(m)
        i = len(p) - 1
        bad_b = p[:i] + bytes([p[i] ^ 0xFF]) + p[i + 1 :]
        strict += int(unpad(bad_b) is None)
    return {
        "synthetic_roundtrip": float(rt / n),
        "synthetic_bad_pad_reject": float(strict / n),
    }
