"""FrodoKEM-style KEM — plain (unstructured) LWE over Z_q matrices.

No ring structure: A uniform Z_q^{n x n}, secrets small-Gaussian. Ephemeral
encapsulation B' = s'A + e', v = s'B + e'' + Encode(mu); decap rounds
v - s B' back to the message. Toy n=16 keeps sampling cheap; algebra is
the real scheme's.
"""

import hashlib

import numpy as np

_SEED = 20261231 + 869

N, Q = 16, 1 << 15


def _h(*parts: bytes) -> bytes:
    m = hashlib.sha256()
    for p in parts:
        m.update(p)
    return m.digest()


def _err(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(-2, 3, (N, N))


def _encode(mu: int) -> np.ndarray:
    return np.full((N, N), (mu & 1) * (Q // 2), dtype=np.int64)


def keygen(seed: bytes) -> tuple[tuple[bytes, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(int.from_bytes(_h(seed, b"fk"), "little") % (1 << 31))
    s = _err(rng)
    e = _err(rng)
    rngA = np.random.default_rng(int.from_bytes(_h(seed, b"A"), "little") % (1 << 31))
    A = rngA.integers(0, Q, (N, N))
    b = (A @ s + e) % Q
    return (seed, b), s


def encap(pk: tuple[bytes, np.ndarray], seed: bytes) -> tuple[tuple[np.ndarray, np.ndarray], bytes]:
    seedA, b = pk
    rng = np.random.default_rng(int.from_bytes(_h(seed, b"fe"), "little") % (1 << 31))
    sp = _err(rng)
    ep = _err(rng)
    epp = _err(rng)
    rngA = np.random.default_rng(int.from_bytes(_h(seedA, b"A"), "little") % (1 << 31))
    A = rngA.integers(0, Q, (N, N))
    bp = (sp @ A + ep) % Q
    mu = _h(seed, b"mu")[0] & 1
    v = (sp @ b + epp + _encode(mu)) % Q
    ct = (bp, v)
    ss = _h(bytes([mu]), _h(bp.tobytes(), v.tobytes()))
    return ct, ss


def decap(sk: np.ndarray, ct: tuple[np.ndarray, np.ndarray]) -> bytes:
    bp, v = ct
    w = (v - bp @ sk) % Q
    mu = int(round(float(w[0, 0]) / (Q // 2)) % 2)
    return _h(bytes([mu]), _h(bp.tobytes(), v.tobytes()))


def bench_frodokem(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: shared-secret agreement; tampered ct -> different key."""
    rng = np.random.default_rng(seed)
    agree, tamper = 0, 0
    for _ in range(8):
        pk, sk = keygen(rng.bytes(16))
        ct, ss = encap(pk, rng.bytes(16))
        agree += int(decap(sk, ct) == ss)
        bad = (ct[0], (ct[1] + Q // 2) % Q)
        tamper += int(decap(sk, bad) != ss)
    return {"synthetic_frodokem": 1.0 if (agree == 8 and tamper == 8) else agree / 8.0}
