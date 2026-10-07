"""BIKE-lite: QC-MDPC-style code KEM with Gallager bit-flipping decoding (SYNTHETIC).

Ring R = F2[x]/(x^r - 1). Secret h0, h1 sparse; public h = h1 * h0^{-1}.
Ciphertext c = e0 + e1 * h; syndrome s = c * h0 = e0*h0 + e1*h1 is decoded by
bit-flipping on the sparse parity checks.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 939


def poly_mul(a: np.ndarray, b: np.ndarray, r: int) -> np.ndarray:
    out = np.zeros(r, dtype=np.uint8)
    for i in np.flatnonzero(a):
        for j in np.flatnonzero(b):
            out[(int(i) + int(j)) % r] ^= 1
    return out


def poly_inv(a: np.ndarray, r: int) -> np.ndarray | None:
    """Inverse in F2[x]/(x^r-1) via circulant linear solve, if it exists."""
    m = np.zeros((r, r), dtype=np.uint8)
    for i in range(r):
        for j in np.flatnonzero(a):
            m[(int(j) + i) % r, i] ^= 1
    aug = np.concatenate([m, np.eye(r, dtype=np.uint8)], axis=1)
    for c in range(r):
        p = next((k for k in range(c, r) if aug[k, c]), None)
        if p is None:
            return None
        if p != c:
            aug[[c, p]] = aug[[p, c]]
        for k in range(r):
            if k != c and aug[k, c]:
                aug[k] ^= aug[c]
    return aug[:, r].astype(np.uint8)


def _sparse(rng: np.random.Generator, r: int, w: int) -> np.ndarray:
    out = np.zeros(r, dtype=np.uint8)
    out[rng.choice(r, w, replace=False)] = 1
    return out


def bitflip_decode(
    s: np.ndarray, h0: np.ndarray, h1: np.ndarray, r: int, iters: int = 100
) -> tuple[np.ndarray, np.ndarray] | None:
    """Recover sparse (e0, e1) from syndrome s = e0*h0 + e1*h1.

    Gallager-B: each round flips every variable node touching the maximal
    number of unsatisfied checks; check j of block i is the circulant column
    roll(h_i, j).
    """
    e0 = np.zeros(r, dtype=np.uint8)
    e1 = np.zeros(r, dtype=np.uint8)
    cur = s.astype(np.uint8).copy()
    checks0 = np.array([np.roll(h0, i) for i in range(r)], dtype=np.uint8)
    checks1 = np.array([np.roll(h1, i) for i in range(r)], dtype=np.uint8)
    for _ in range(iters):
        if not cur.any():
            return e0, e1
        cnt0 = checks0 @ cur
        cnt1 = checks1 @ cur
        thr = int(max(cnt0.max(), cnt1.max()))
        if thr == 0:
            return None
        flips0 = np.flatnonzero(cnt0 == thr)
        flips1 = np.flatnonzero(cnt1 == thr)
        e0[flips0] ^= 1
        e1[flips1] ^= 1
        cur ^= checks0[flips0].sum(axis=0) % 2
        cur ^= checks1[flips1].sum(axis=0) % 2
    return (e0, e1) if not cur.any() else None


def keygen(rng: np.random.Generator, r: int, w: int) -> dict[str, np.ndarray] | None:
    h0 = _sparse(rng, r, w)
    h1 = _sparse(rng, r, w)
    h0i = poly_inv(h0, r)
    if h0i is None:
        return None
    return {"h0": h0, "h1": h1, "h": poly_mul(h1, h0i, r)}


def encaps(
    pk_h: np.ndarray, rng: np.random.Generator, r: int, t: int
) -> tuple[np.ndarray, np.ndarray]:
    e0 = _sparse(rng, r, t // 2)
    e1 = _sparse(rng, r, t - t // 2)
    c = poly_mul(e1, pk_h, r) ^ e0
    return c, np.concatenate([e0, e1])


def decaps(kp: dict[str, np.ndarray], c: np.ndarray, r: int) -> np.ndarray | None:
    s = poly_mul(c, kp["h0"], r)
    dec = bitflip_decode(s, kp["h0"], kp["h1"], r)
    if dec is None:
        return None
    return np.concatenate(dec)


def bench_bike_lite(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    r, w, t = 61, 7, 3
    kp = None
    for _ in range(300):
        kp = keygen(rng, r, w)
        if kp is not None:
            break
    if kp is None:
        raise RuntimeError("no invertible h0 found")
    ok = 0
    trials = 30
    for _ in range(trials):
        c, e = encaps(kp["h"], rng, r, t)
        d = decaps(kp, c, r)
        if d is not None and np.array_equal(d, e):
            ok += 1
    h_ok = int(np.array_equal(poly_mul(kp["h"], kp["h0"], r), kp["h1"]))
    return {"synthetic_bike_lite": 0.85 * (ok / trials) + 0.15 * h_ok}
