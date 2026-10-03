"""HQC-lite: Hamming Quasi-Cyclic KEM structure over R = F2[x]/(x^r - 1).

pk (h, s = x + h*y) with x, y sparse; encryption u = r1 + r2*h,
v = m*G + s*r2 + e; decryption C.decode(v - trunc(u*y)) — the error term
x*r2 - r1*y + e stays below the concatenated code's radius.
Code = RM(1,3) [8,4,4] inner + length-3 repetition outer (HQC-style concat).
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 940

_RM_G = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1],
    ],
    dtype=np.uint8,
)


def poly_mul(a: np.ndarray, b: np.ndarray, r: int) -> np.ndarray:
    out = np.zeros(r, dtype=np.uint8)
    for i in np.flatnonzero(a):
        for j in np.flatnonzero(b):
            out[(int(i) + int(j)) % r] ^= 1
    return out


def _sparse(rng: np.random.Generator, r: int, w: int) -> np.ndarray:
    out = np.zeros(r, dtype=np.uint8)
    out[rng.choice(r, w, replace=False)] = 1
    return out


def concat_encode(m: np.ndarray, n1: int = 5) -> np.ndarray:
    """4-bit message -> RM(1,3) [8,4,4] -> each bit repeated n1 -> 24 bits."""
    cw = (m.reshape(1, -1) @ _RM_G % 2).ravel().astype(np.uint8)
    return np.repeat(cw, n1)


def _rm_decode(cw8: np.ndarray) -> np.ndarray:
    """Majority-logic RM(1,3) decode: recover (a0,a1,a2,a3)."""
    best = (9, np.zeros(4, dtype=np.uint8))
    for k in range(16):
        m = np.array([(k >> i) & 1 for i in range(4)], dtype=np.uint8)
        c = (m.reshape(1, -1) @ _RM_G % 2).ravel()
        d = int(np.sum(c != cw8))
        if d < best[0]:
            best = (d, m)
    return best[1]


def concat_decode(v: np.ndarray, n1: int = 5) -> np.ndarray:
    cw8 = np.array(
        [1 if v[i * n1 : (i + 1) * n1].sum() * 2 > n1 else 0 for i in range(8)],
        dtype=np.uint8,
    )
    return _rm_decode(cw8)


def keygen(rng: np.random.Generator, r: int, w: int) -> dict[str, np.ndarray]:
    x = _sparse(rng, r, w)
    y = _sparse(rng, r, w)
    h = rng.integers(0, 2, r).astype(np.uint8)
    s = x ^ poly_mul(h, y, r)
    return {"x": x, "y": y, "h": h, "s": s}


def encaps(
    kp: dict[str, np.ndarray],
    rng: np.random.Generator,
    r: int,
    wt: int,
    n1: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r1 = _sparse(rng, r, wt)
    r2 = _sparse(rng, r, wt)
    e = _sparse(rng, r, 1)
    u = r1 ^ poly_mul(r2, kp["h"], r)
    sr2 = poly_mul(kp["s"], r2, r)[: 8 * n1]
    m = (rng.integers(0, 2, 4)).astype(np.uint8)
    e_short = np.zeros(8 * n1, dtype=np.uint8)
    e_short[: min(8 * n1, r)] = e[: min(8 * n1, r)]
    v = concat_encode(m, n1) ^ sr2 ^ e_short
    return u, v, m


def decaps(
    kp: dict[str, np.ndarray], u: np.ndarray, v: np.ndarray, r: int, n1: int = 5
) -> np.ndarray:
    w = v ^ poly_mul(u, kp["y"], r)[: 8 * n1]
    return concat_decode(w, n1)


def bench_hqc_lite(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    r, w, wt, n1 = 61, 2, 2, 5
    kp = keygen(rng, r, w)
    ok = 0
    trials = 30
    for _ in range(trials):
        u, v, m = encaps(kp, rng, r, wt, n1)
        if np.array_equal(decaps(kp, u, v, r, n1), m):
            ok += 1
    s_ok = int(np.array_equal(kp["s"], kp["x"] ^ poly_mul(kp["h"], kp["y"], r)))
    cw = concat_encode(np.array([1, 0, 1, 1], dtype=np.uint8), n1)
    cw[3] ^= 1
    rm_ok = int(np.array_equal(concat_decode(cw, n1), [1, 0, 1, 1]))
    return {"synthetic_hqc_lite": 0.7 * (ok / trials) + 0.15 * s_ok + 0.15 * rm_ok}
