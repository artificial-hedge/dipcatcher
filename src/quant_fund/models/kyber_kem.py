"""ML-KEM / Kyber-style KEM — module-LWE key encapsulation over R_q[x]/(x^n+1) (SYNTHETIC).

Kyber-512-shaped parameters (n=256, q=3329, k=2) with the Fujisaki-Okamoto
transform: keypair -> encap -> decap shares a 32-byte key; a tampered
ciphertext produces implicit rejection (different key, no failure flag).
"""

import hashlib

import numpy as np

_SEED = 20261231 + 867

N, Q, K = 256, 3329, 2


def _h(*parts: bytes) -> bytes:
    m = hashlib.sha256()
    for p in parts:
        m.update(p)
    return m.digest()


def _uniform(seed: bytes, i: int, j: int) -> np.ndarray:
    rng = np.random.default_rng(int.from_bytes(_h(seed, bytes([i, j])), "little") % (1 << 31))
    return np.asarray(rng.integers(0, Q, N))


def _cbd(rng: np.random.Generator, eta: int = 1) -> np.ndarray:
    a = rng.integers(0, 2, N)
    b = rng.integers(0, 2, N)
    for _ in range(eta - 1):
        a = a + rng.integers(0, 2, N)
        b = b + rng.integers(0, 2, N)
    return a - b


def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = np.zeros(N, dtype=np.int64)
    for i in range(N):
        for j in range(N):
            s = 1 if i + j < N else -1
            c[(i + j) % N] = (c[(i + j) % N] + s * int(a[i]) * int(b[j])) % Q
    return c


def _add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray((a + b) % Q)


def _sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray((a - b) % Q)


def _encode(m: bytes) -> np.ndarray:
    """Message bits -> {0, floor(q/2)} coefficients (Decompress_q(1))."""
    bits = np.unpackbits(np.frombuffer(m[:32], dtype=np.uint8)).astype(np.int64)
    return (bits * ((Q + 1) // 2)) % Q


def _decode(v: np.ndarray) -> bytes:
    """Compress to 1 bit per coefficient."""
    bits = (v * 2 + Q // 2) // Q % 2
    return np.packbits(bits.astype(np.uint8)).tobytes()


def keygen(seed: bytes) -> tuple[tuple[bytes, list[np.ndarray]], np.ndarray]:
    rng = np.random.default_rng(int.from_bytes(_h(seed, b"kg"), "little") % (1 << 31))
    seedA = seed[:16]
    s = [_cbd(rng) for _ in range(K)]
    e = [_cbd(rng) for _ in range(K)]
    t = [np.zeros(N, dtype=np.int64) for _ in range(K)]
    for i in range(K):
        for j in range(K):
            t[i] = _add(t[i], _mul(_uniform(seedA, i, j), s[j]))
        t[i] = _add(t[i], e[i])
    return (seedA, t), np.array(s)


def encap(
    pk: tuple[bytes, list[np.ndarray]], seed: bytes
) -> tuple[tuple[np.ndarray, np.ndarray], bytes]:
    seedA, t = pk
    m = _h(seed, b"m")[:32]
    rng = np.random.default_rng(int.from_bytes(_h(seed, b"cp"), "little") % (1 << 31))
    r = [_cbd(rng) for _ in range(K)]
    e1 = [_cbd(rng) for _ in range(K)]
    e2 = _cbd(rng)
    u = [np.zeros(N, dtype=np.int64) for _ in range(K)]
    for i in range(K):
        for j in range(K):
            u[i] = _add(u[i], _mul(_uniform(seedA, j, i), r[j]))
        u[i] = _add(u[i], e1[i])
    v = np.zeros(N, dtype=np.int64)
    for j in range(K):
        v = _add(v, _mul(t[j], r[j]))
    v = _add(_add(v, e2), _encode(m))
    c = (np.array(u), v)
    ss = _h(m, _h(c[0].tobytes(), c[1].tobytes()))
    return c, ss


def decap(sk: np.ndarray, c: tuple[np.ndarray, np.ndarray]) -> bytes:
    u, v = c
    w = np.zeros(N, dtype=np.int64)
    for j in range(K):
        w = _add(w, _mul(sk[j], u[j]))
    m = _decode(_sub(v, w))
    return _h(m, _h(u.tobytes(), v.tobytes()))


def bench_kyber_kem(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: shared-secret agreement; tampered ct -> different key."""
    rng = np.random.default_rng(seed)
    agree, tamper = 0, 0
    for _ in range(6):
        pk, sk = keygen(rng.bytes(16))
        c, ss = encap(pk, rng.bytes(16))
        agree += int(decap(sk, c) == ss)
        bad = (c[0].copy(), (c[1] + 1) % Q)
        tamper += int(decap(sk, bad) != ss)
    return {"synthetic_kyber_kem": 1.0 if (agree == 6 and tamper == 6) else agree / 6.0}
