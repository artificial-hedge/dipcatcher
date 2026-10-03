"""Dupire local-volatility extraction from a synthetic call surface.

Build C(K,T) by Crank-Nicolson on the Dupire forward equation
C_T = 0.5 sigma_loc(K)^2 K^2 C_KK (r=0) under a known
sigma_loc(K) = a + b tanh((K-100)/20), then invert
sigma^2 = 2 C_T / (K^2 C_KK) and compare on the interior.
"""

import numpy as np
from scipy.linalg import solve_banded


def _call_surface(sig: np.ndarray, ks: np.ndarray, ts: np.ndarray, s0: float = 100.0) -> np.ndarray:
    nt, nk = len(ts), len(ks)
    c = np.zeros((nt, nk))
    c[0] = np.maximum(s0 - ks, 0.0)
    dt = ts[1] - ts[0]
    dk = ks[1] - ks[0]
    beta = 0.5 * dt * 0.5 * sig**2 * ks**2 / dk**2  # per grid point
    m = nk - 2
    bi = beta[1:-1]
    bands = np.zeros((3, m))
    bands[0, 1:] = -bi[:-1]  # upper diag (row 0, cols 1..m-1)
    bands[1] = 1.0 + 2.0 * bi
    bands[2, :-1] = -bi[1:]  # lower diag
    for it in range(1, nt):
        p = c[it - 1]
        rhs = bi * p[:-2] + (1.0 - 2.0 * bi) * p[1:-1] + bi * p[2:]
        c[it, 1:-1] = solve_banded((1, 1), bands, rhs)
        c[it, 0] = max(s0 - ks[0], 0.0)
        c[it, -1] = 0.0
        c[it] = np.maximum(c[it], np.maximum(s0 - ks, 0.0))
    return c


def bench_dupire_localvol(seed: int = 5801) -> dict[str, float]:
    _ = seed
    ks = np.linspace(60, 160, 161)
    ts = np.linspace(0.0, 0.6, 600)
    sig_fn = lambda k: 0.2 + 0.08 * np.tanh((k - 100.0) / 20.0)  # noqa: E731
    sig_true = sig_fn(ks)
    c = _call_surface(sig_true, ks, ts)
    dk = ks[1] - ks[0]
    dt = ts[1] - ts[0]
    ct = np.gradient(c, dt, axis=0)
    ck2 = np.gradient(np.gradient(c, dk, axis=1), dk, axis=1)
    sig2 = 2.0 * ct / np.maximum(ks[None, :] ** 2 * ck2, 1e-10)
    i0, i1 = 200, 500
    j0, j1 = 50, 110
    est = np.sqrt(np.maximum(sig2[i0:i1, j0:j1], 0))
    tru = sig_true[j0:j1]
    good = np.isfinite(est).all(0) & (est.mean(0) > 0.02) & (tru > 0.05)
    err = float(np.mean(np.abs(est.mean(0)[good] - tru[good]) / tru[good]))
    return {
        "synthetic_dupire_relerr": err,
        "synthetic_dupire_est_mid": float(est.mean(0)[40]),
        "synthetic_dupire_true_mid": float(tru[40]),
        "synthetic_dupire_ncell": float(good.sum()),
    }
