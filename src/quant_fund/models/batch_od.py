"""Batch least-squares orbit determination from range measurements.

SYNTHETIC bench: simulates a two-body orbit with ground-station ranges,
perturbs the initial state, and recovers it via Gauss-Newton on the
state-transition-matrix linearization (numerical STM).
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm

_SEED = 20261231 + 922
_MU = 398600.4418


def _accel(r: np.ndarray, mu: float) -> np.ndarray:
    rm = float(norm(r))
    return -mu * r / rm**3


def _prop_full(y: np.ndarray, dt: float, mu: float, nstep: int = 200) -> np.ndarray:
    """Propagate 6-state with 6x6 STM (36 extra components)."""

    def f(yy: np.ndarray) -> np.ndarray:
        out = np.zeros(42)
        out[:3] = yy[3:6]
        out[3:6] = _accel(yy[:3], mu)
        rm = float(norm(yy[:3]))
        # gravity gradient d(accel)/dr = -mu/r^3 I + 3mu r r^T / r^5
        G = -mu / rm**3 * np.eye(3) + 3 * mu * np.outer(yy[:3], yy[:3]) / rm**5
        Phi = yy[6:].reshape(6, 6)
        A = np.zeros((6, 6))
        A[:3, 3:6] = np.eye(3)
        A[3:6, :3] = G
        out[6:] = (A @ Phi).ravel()
        return out

    h = dt / nstep
    for _ in range(nstep):
        k1 = f(y)
        k2 = f(y + h / 2 * k1)
        k3 = f(y + h / 2 * k2)
        k4 = f(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def batch_od(
    times: np.ndarray,
    site: np.ndarray,
    ranges: np.ndarray,
    y0: np.ndarray,
    mu: float = _MU,
    itmax: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Newton estimation of the epoch state from station ranges."""
    y0 = np.asarray(y0, dtype=np.float64).copy()
    site = np.asarray(site, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    ranges = np.asarray(ranges, dtype=np.float64)
    cov = np.eye(6)
    lam = 1e-6

    def cost(y: np.ndarray) -> float:
        c = 0.0
        for i, t in enumerate(times):
            yy = _prop_full(np.concatenate([y, np.eye(6).ravel()]), t, mu, nstep=60)
            c += (float(norm(yy[:3] - site)) - float(ranges[i])) ** 2
        return float(c)

    c0 = cost(y0)
    for _ in range(itmax):
        H = np.zeros((len(times), 6))
        res = np.zeros(len(times))
        for i, t in enumerate(times):
            y = _prop_full(np.concatenate([y0, np.eye(6).ravel()]), t, mu, nstep=60)
            r = y[:3]
            Phi = y[6:].reshape(6, 6)
            los = r - site
            rng_i = float(norm(los))
            res[i] = ranges[i] - rng_i
            H[i, :] = (los / rng_i) @ Phi[:3, :]
        JTJ = H.T @ H
        JTr = H.T @ res
        # column equilibration keeps the covariance solve well-conditioned
        col = np.sqrt(np.diag(JTJ))
        col[col == 0] = 1.0
        D = np.diag(1.0 / col)
        Jn = D @ JTJ @ D
        try:
            cov = np.asarray(D @ np.linalg.inv(Jn) @ D, dtype=np.float64)
        except np.linalg.LinAlgError:
            cov = np.eye(6)
        # Levenberg damping: scale step until cost decreases
        stepped = False
        for _inner in range(12):
            dy = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ) + 1e-12), JTr)
            c1 = cost(y0 + dy)
            if c1 < c0:
                y0 = y0 + dy
                c0 = c1
                lam = max(lam / 10.0, 1e-12)
                stepped = True
                break
            lam *= 10.0
        if not stepped or float(norm(dy)) < 1e-9:
            break
    return y0, cov


def bench_batch_od(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    r0 = np.asarray([7000.0, 0.0, 0.0])
    v0 = np.asarray([0.0, np.sqrt(_MU / 7000.0) * np.cos(0.4), np.sqrt(_MU / 7000.0) * np.sin(0.4)])
    truth = np.concatenate([r0, v0])
    site = np.asarray([6378.0, 0.0, 0.0])
    times = np.linspace(-1500.0, 1500.0, 13)
    ranges = np.zeros(len(times))
    for i, t in enumerate(times):
        y = _prop_full(np.concatenate([truth, np.eye(6).ravel()]), t, _MU, nstep=60)
        ranges[i] = norm(y[:3] - site)
    # perturb initial guess
    guess = truth + np.array([30.0, -20.0, 10.0, 0.05, -0.03, 0.02])
    est, cov = batch_od(times, site, ranges, guess)
    perr = float(norm(est[:3] - r0))
    verr = float(norm(est[3:] - v0))
    score += 1.0 if perr < 1.0 else 0.0
    score += 1.0 if verr < 0.1 else 0.0
    # residuals after fit tiny
    res = np.zeros(len(times))
    for i, t in enumerate(times):
        y = _prop_full(np.concatenate([est, np.eye(6).ravel()]), t, _MU, nstep=60)
        res[i] = ranges[i] - norm(y[:3] - site)
    score += 1.0 if float(np.abs(res).max()) < 0.1 else 0.0
    # GN contraction: final error far below the injected 42-km/0.08-kms offset
    init_err = float(norm(guess - truth))
    score += 1.0 if float(norm(est - truth)) < 0.05 * init_err else 0.0
    rng.random()
    return {"synthetic_batch_od": score / 4.0}
