"""McEliece-lite: code-based PKE over a Hamming [15,11,3] code.

Public key G = S * G0 * P (scramble + column permutation); encryption adds a
weight-t error; decryption un-permutes, syndrome-decodes, un-scrambles.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 938


def _gf2(m: np.ndarray) -> np.ndarray:
    return np.asarray(m, dtype=np.uint8) % 2


def _mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.uint64) @ b.astype(np.uint64) % 2).astype(np.uint8)


def _build_gh() -> tuple[np.ndarray, np.ndarray]:
    """[15,11,3] Hamming code: H columns = all nonzero 4-bit vectors.

    Info columns carry vectors of weight >= 2; the four weight-1 columns are
    parity positions. Row r of G has its info bit at column c plus the parity
    columns whose syndromes XOR to syndrome(c) = c+1.
    """
    h = np.zeros((4, 15), dtype=np.uint8)
    for i in range(15):
        for b in range(4):
            h[b, i] = (i + 1) >> b & 1
    info_cols = [i for i in range(15) if bin(i + 1).count("1") >= 2]
    g = np.zeros((11, 15), dtype=np.uint8)
    for r, c in enumerate(info_cols):
        g[r, c] = 1
        v = c + 1
        for b in range(4):
            if v >> b & 1:
                g[r, (1 << b) - 1] ^= 1
    return g, h


def _det(m: np.ndarray) -> int:
    a = _gf2(m.copy())
    n = a.shape[0]
    for c in range(n):
        p = next((r for r in range(c, n) if a[r, c]), None)
        if p is None:
            return 0
        if p != c:
            a[[c, p]] = a[[p, c]]
        for r in range(c + 1, n):
            if a[r, c]:
                a[r] ^= a[c]
    return 1


def _rand_inv(rng: np.random.Generator, n: int) -> np.ndarray:
    while True:
        m = _gf2(rng.integers(0, 2, (n, n)))
        if _det(m) == 1:
            return m


def _inv(m: np.ndarray) -> np.ndarray:
    a = _gf2(m.copy())
    n = a.shape[0]
    inv = np.eye(n, dtype=np.uint8)
    for c in range(n):
        p = next(r for r in range(c, n) if a[r, c])
        if p != c:
            a[[c, p]] = a[[p, c]]
            inv[[c, p]] = inv[[p, c]]
        for r in range(n):
            if r != c and a[r, c]:
                a[r] ^= a[c]
                inv[r] ^= inv[c]
    return inv


def keygen(rng: np.random.Generator) -> dict[str, np.ndarray]:
    g0, h0 = _build_gh()
    s = _rand_inv(rng, 11)
    perm = rng.permutation(15)
    p = np.eye(15, dtype=np.uint8)[:, perm]
    return {"G": _mm(_mm(s, g0), p), "S": s, "perm": perm, "H0": h0}


def encrypt(pk: np.ndarray, m: np.ndarray, e: np.ndarray) -> np.ndarray:
    return _gf2(_mm(m.reshape(1, -1), pk).ravel() + e)


def decrypt(kp: dict[str, np.ndarray], c: np.ndarray) -> np.ndarray:
    c0 = _gf2(c)[np.argsort(kp["perm"])]
    syn = _mm(kp["H0"], c0.reshape(-1, 1)).ravel()
    err = int(syn[0] + 2 * syn[1] + 4 * syn[2] + 8 * syn[3])
    if err:
        c0[err - 1] ^= 1
    info_cols = [i for i in range(15) if bin(i + 1).count("1") >= 2]
    m0 = c0[info_cols]
    return _mm(m0.reshape(1, -1), _inv(kp["S"])).ravel()


def bench_mceliece_lite(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    kp = keygen(rng)
    h_pub = kp["H0"][:, kp["perm"]]
    code_ok = int(np.all(_mm(kp["G"], h_pub.T) == 0))
    ok = 0
    trials = 60
    for _ in range(trials):
        m = _gf2(rng.integers(0, 2, 11))
        e = np.zeros(15, dtype=np.uint8)
        e[int(rng.integers(0, 15))] = 1
        if np.array_equal(decrypt(kp, encrypt(kp["G"], m, e)), m):
            ok += 1
    return {"synthetic_mceliece_lite": 0.8 * (ok / trials) + 0.2 * code_ok}
