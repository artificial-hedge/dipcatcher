"""ML-DSA / Dilithium-style signatures — Fiat-Shamir with aborts over R_q (SYNTHETIC).

Module structure: keygen t = A s1 + s2; sign samples y, sets w = Ay, hashes
c = H(mu || HighBits(w)), z = y + c s1 with rejection sampling; verify checks
|z| bound and c == H(mu || HighBits(Az - ct)). Toy-grade parameters (n=32,
k=l=2) keep the algebra identical to the standard while staying fast.
"""

import hashlib

import numpy as np

_SEED = 20261231 + 868

N, Q, K, L = 32, 8380417, 2, 2
ETA, TAU = 1, 8
GAMMA1, GAMMA2 = 1 << 14, 512
BETA = TAU * ETA


def _h(*parts: bytes) -> bytes:
    m = hashlib.sha256()
    for p in parts:
        m.update(p)
    return m.digest()


def _uniform(seed: bytes, i: int, j: int) -> np.ndarray:
    rng = np.random.default_rng(int.from_bytes(_h(seed, bytes([i, j])), "little") % (1 << 31))
    return rng.integers(0, Q, N)


def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = np.zeros(N, dtype=np.int64)
    for i in range(N):
        for j in range(N):
            s = 1 if i + j < N else -1
            c[(i + j) % N] = (c[(i + j) % N] + s * int(a[i]) * int(b[j])) % Q
    return c


def _small_poly(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(-ETA, ETA + 1, N)


def _challenge(seed_c: bytes) -> np.ndarray:
    rng = np.random.default_rng(int.from_bytes(seed_c[:4], "little"))
    c = np.zeros(N, dtype=np.int64)
    c[rng.choice(N, TAU, replace=False)] = rng.choice([-1, 1], TAU)
    return c


def _highbits(w: np.ndarray) -> np.ndarray:
    return (w % Q) // (2 * GAMMA2)


def _matvec(A_seed: bytes, v: np.ndarray, transposed: bool = False) -> np.ndarray:
    out = np.zeros((v.shape[0], N), dtype=np.int64)
    for i in range(v.shape[0]):
        for j in range(v.shape[0]):
            a = _uniform(A_seed, j, i) if transposed else _uniform(A_seed, i, j)
            out[i] = (out[i] + _mul(a, v[j])) % Q
    return out


def keygen(seed: bytes) -> tuple[tuple[bytes, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(int.from_bytes(_h(seed, b"dk"), "little") % (1 << 31))
    s1 = np.array([_small_poly(rng) for _ in range(L)])
    s2 = np.array([_small_poly(rng) for _ in range(K)])
    t = (_matvec(seed, s1) + s2) % Q
    return (seed, t), np.concatenate([s1, s2])


def sign(
    sk: np.ndarray, pk: tuple[bytes, np.ndarray], msg: bytes, rng: np.random.Generator
) -> np.ndarray | None:
    seedA, t = pk
    s1 = sk[:L]
    mu = _h(t.tobytes(), msg)
    for _ in range(200):
        y = rng.integers(-GAMMA1 + BETA, GAMMA1 - BETA + 1, (L, N))
        w = _matvec(seedA, y)
        c = _challenge(_h(mu, _highbits(w).tobytes()))
        z = (y + np.array([_mul(c, p) for p in s1])) % Q
        if np.abs((z + Q // 2) % Q - Q // 2).max() >= GAMMA1 - BETA:
            continue
        w_check = (_matvec(seedA, z) - np.array([_mul(c, p) for p in t])) % Q
        if not np.array_equal(_highbits(w_check), _highbits(w)):
            continue
        return np.concatenate([z.ravel(), np.frombuffer(c.tobytes(), dtype=np.int64)])
    return None


def verify(pk: tuple[bytes, np.ndarray], msg: bytes, sig: np.ndarray) -> bool:
    seedA, t = pk
    z = sig[: L * N].reshape(L, N) % Q
    c = sig[L * N :].astype(np.int64)
    if np.abs((z + Q // 2) % Q - Q // 2).max() >= GAMMA1 - BETA:
        return False
    mu = _h(t.tobytes(), msg)
    w1 = (_matvec(seedA, z) - np.array([_mul(c, p) for p in t])) % Q
    return bool(np.array_equal(c, _challenge(_h(mu, _highbits(w1).tobytes()))))


def bench_dilithium_sig(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: valid sigs verify; wrong message or forged z fails."""
    rng = np.random.default_rng(seed)
    pk, sk = keygen(rng.bytes(16))
    good, forge = 0, 0
    for _ in range(5):
        msg = rng.bytes(20)
        sig = sign(sk, pk, msg, rng)
        good += int(sig is not None and verify(pk, msg, sig))
        bad = sig.copy() if sig is not None else np.zeros(L * N + N, dtype=np.int64)
        if sig is not None:
            bad[0] = (bad[0] + 1) % Q
            forge += int(not verify(pk, msg, bad))
    return {"synthetic_dilithium_sig": 1.0 if (good == 5 and forge == 5) else good / 5.0}
