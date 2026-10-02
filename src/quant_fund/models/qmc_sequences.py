"""Quasi-Monte Carlo sequence canon.

Low-discrepancy point generators for the unit cube — each
deterministic, uniform, and dramatically better-equidistributed
than i.i.d. MC at moderate n:

- ``halton`` — Halton sequence via radical inverses in the
  first d primes.
- ``hammersley`` — Hammersley set (n/d along the first axis,
  Halton in the rest).
- ``faure`` — Faure sequence in base b = smallest prime >= d
  with the (a_i · S_k) digit recursion.
- ``korobov`` — Korobov/rank-1 lattice x_i = i·alpha mod 1 with
  irrational alpha (Fibonacci-like lattice).
- ``l2_discrepancy`` — Warnock's closed-form L2 discrepancy.
- ``mc_uniform`` — i.i.d. MC baseline.

Bench: L2 discrepancy ordering + quadrature error on a smooth
integrand vs MC (SYNTHETIC only).
"""

from __future__ import annotations

import math

import numpy as np

FloatArray = np.ndarray


def _primes(k: int) -> list[int]:
    out: list[int] = []
    c = 2
    while len(out) < k:
        if all(c % p for p in out if p * p <= c):
            out.append(c)
        c += 1
    return out


def _radical_inverse(i: int, base: int) -> float:
    f, r, v = 1.0 / base, i, 0.0
    while r > 0:
        v += f * (r % base)
        r //= base
        f /= base
    return v


def halton(n: int, d: int) -> FloatArray:
    ps = _primes(d)
    return np.array([[_radical_inverse(i + 1, p) for p in ps] for i in range(n)])


def hammersley(n: int, d: int) -> FloatArray:
    ps = _primes(d - 1) if d > 1 else []
    pts = np.empty((n, d))
    for i in range(n):
        pts[i, 0] = i / n
        pts[i, 1:] = [_radical_inverse(i + 1, p) for p in ps]
    return pts


def faure(n: int, d: int) -> FloatArray:
    """Faure sequence (no scrambling), base b = prime >= d."""
    b = _primes(d)[-1]
    pts = np.empty((n, d))
    for i in range(n):
        digits: list[int] = []
        t = i + 1
        while t > 0:
            digits.append(t % b)
            t //= b
        pts[i, 0] = _radical_inverse(i + 1, b)
        for j in range(1, d):
            dj = np.array(digits, dtype=np.int64)
            for _ in range(j):
                c = np.zeros(len(dj), dtype=np.int64)
                for i_ in range(len(dj)):
                    for k_ in range(i_, len(dj)):
                        c[i_] += math.comb(k_, i_) * dj[k_]
                dj = c % b
            v, f = 0.0, 1.0 / b
            for dg in dj:
                v += f * int(dg)
                f /= b
            pts[i, j] = v
    return pts


def korobov(n: int, d: int, seed: int = 0) -> FloatArray:
    rng = np.random.default_rng(seed)
    alpha = rng.random(d) * (np.sqrt(5.0) - 1.0) / 2.0 + 0.1
    i = np.arange(1, n + 1)[:, None]
    return np.asarray((i * alpha) % 1.0)


def mc_uniform(n: int, d: int, seed: int = 0) -> FloatArray:
    return np.random.default_rng(seed).random((n, d))


def l2_discrepancy(pts: FloatArray) -> float:
    """Warnock's L2 discrepancy, anchored at the upper corner."""
    pts = np.asarray(pts, dtype=np.float64)
    n, d = pts.shape
    t1 = (1.0 / 3.0) ** d
    t2 = (2.0 ** (1 - d) / n) * float(np.sum(np.prod(1.0 - pts**2, axis=1)))
    mx = np.maximum(pts[:, None, :], pts[None, :, :])
    t3 = (1.0 / (n * n)) * float(np.sum(np.prod(1.0 - mx, axis=2)))
    return float(np.sqrt(max(t1 - t2 + t3, 0.0)))


def bench_qmc(seed: int = 0) -> dict[str, float]:
    n, d = 625, 4
    # int_[0,1]^d prod(cos(x_k)) = sin(1)^d
    exact = float(np.sin(1.0) ** d)
    f = lambda p: float(np.prod(np.cos(p), axis=1).mean() - exact)  # noqa: E731
    pts_h = halton(n, d)
    pts_hm = hammersley(n, d)
    pts_f = faure(n, d)
    pts_k = korobov(n, d, seed=seed)
    pts_mc = mc_uniform(n, d, seed=seed)
    disc_h = l2_discrepancy(pts_h)
    disc_hm = l2_discrepancy(pts_hm)
    disc_f = l2_discrepancy(pts_f)
    disc_mc = l2_discrepancy(pts_mc)
    err_h = abs(f(pts_h))
    err_hm = abs(f(pts_hm))
    err_f = abs(f(pts_f))
    err_k = abs(f(pts_k))
    err_mc = abs(f(pts_mc))
    return {
        "synthetic_disc_halton": disc_h,
        "synthetic_disc_hammersley": disc_hm,
        "synthetic_disc_faure": disc_f,
        "synthetic_disc_mc": disc_mc,
        "synthetic_qerr_halton": float(err_h),
        "synthetic_qerr_hammersley": float(err_hm),
        "synthetic_qerr_faure": float(err_f),
        "synthetic_qerr_korobov": float(err_k),
        "synthetic_qerr_mc": float(err_mc),
    }


__all__ = [
    "bench_qmc",
    "faure",
    "halton",
    "hammersley",
    "korobov",
    "l2_discrepancy",
    "mc_uniform",
]
