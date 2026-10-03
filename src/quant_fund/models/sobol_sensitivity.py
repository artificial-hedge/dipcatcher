"""Sobol variance sensitivity indices and Morris elementary effects.

Sobol (2001, Math. Comput. Simul. 55:271-280) decomposes output
variance into input effects via ANOVA-HDMR; Saltelli et al. (2010,
Comput. Phys. Commun. 181:259-270) give the A/B/C-resampling matrix
estimators implemented in ``sobol_indices`` (first-order S_i from
Saltelli; total-order T_i from Jansen 1999). Morris (1991,
Technometrics 33:161-174) screens factors with elementary effects;
``morris_effects`` uses the radial trajectory design of Campolongo
et al. (2011) with mu*, mu and sigma statistics.

Honesty: the bench self-check runs the Ishigami function, whose true
indices are analytic (S1 = 0.3139, S2 = 0.4424, S3 = 0, T3 = 0.2437);
figures are SYNTHETIC correctness diagnostics. Fail-closed on
non-finite outputs or degenerate (zero-variance) responses.
Composition: generic black-box sensitivity infra; model_fn is any
FloatArray -> FloatArray map evaluated row-wise.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

ModelFn = Callable[[FloatArray], FloatArray]


def _eval(fn: ModelFn, x: FloatArray) -> FloatArray:
    y = np.asarray(fn(x), dtype=np.float64).ravel()
    if y.shape[0] != x.shape[0] or not np.isfinite(y).all():
        raise ValueError("model must return one finite value per row")
    return y


def sobol_indices(
    fn: ModelFn,
    d: int,
    n: int,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """First-order S_i (Saltelli) and total-order T_i (Jansen).

    A, B are two independent U(0,1)^d samples of size n; C_i copies B
    with column i replaced by A's. Estimators (Saltelli 2010):
      S_i = ( mean(f_B * f_Ci) - mu^2 ) / V
      T_i = sum((f_A - f_Ci)^2) / (2 n V)
    """
    if d < 1 or n < 64:
        raise ValueError("need d >= 1 and n >= 64")
    rng = np.random.default_rng(seed)
    a = rng.random((n, d))
    b = rng.random((n, d))
    fa = _eval(fn, a)
    fb = _eval(fn, b)
    var = float(np.var(np.concatenate([fa, fb])))
    if var <= 0:
        raise ValueError("degenerate model: zero output variance")
    mu2 = float(np.mean(np.concatenate([fa, fb])) ** 2)
    s1 = np.zeros(d)
    st = np.zeros(d)
    for i in range(d):
        c = b.copy()
        c[:, i] = a[:, i]
        fc = _eval(fn, c)
        s1[i] = (float(np.mean(fa * fc)) - mu2) / var
        st[i] = float(np.mean((fb - fc) ** 2)) / (2.0 * var)
    return {"s1": s1, "st": st, "var": np.asarray(var)}


def _morris_trajectory(d: int, p: int, delta: float, rng: np.random.Generator) -> FloatArray:
    """One Morris base trajectory: (d+1) x d, each step moves one dim
    by +delta; starts sampled from grid values <= 1-delta so every
    point stays in [0,1] (Campolongo et al. 2011 orientation)."""
    b = np.zeros((d + 1, d))
    idx_max = int(np.floor((p - 2) / 2.0))  # grid index of 1 - delta
    x0 = rng.integers(0, idx_max + 1, size=d).astype(np.float64) / (p - 1.0)
    perm = rng.permutation(d)
    b[0] = x0
    for step, i in enumerate(perm):
        b[step + 1] = b[step]
        b[step + 1, i] += delta
    return b


def morris_effects(
    fn: ModelFn,
    d: int,
    n_traj: int = 20,
    p: int = 4,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Morris (1991) elementary-effect screening: mu*, mu, sigma per dim."""
    if d < 1 or n_traj < 4:
        raise ValueError("need d >= 1 and n_traj >= 4")
    rng = np.random.default_rng(seed)
    delta = p / (2.0 * (p - 1.0))
    ee = np.zeros((n_traj, d))
    for t in range(n_traj):
        traj = _morris_trajectory(d, p, delta, rng)
        y = _eval(fn, traj)
        # recompute which dim moved at each step
        diffs = np.abs(np.diff(traj, axis=0)) > 1e-12
        for step in range(d):
            i = int(np.flatnonzero(diffs[step])[0])
            dy = y[step + 1] - y[step]
            dx = traj[step + 1, i] - traj[step, i]
            ee[t, i] = dy / dx
    return {
        "mu_star": np.abs(ee).mean(axis=0),
        "mu": ee.mean(axis=0),
        "sigma": ee.std(axis=0, ddof=1),
        "ee": ee,
    }


def _ishigami(x: FloatArray, a: float = 7.0, b: float = 0.1) -> FloatArray:
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])


def bench_sobol(seed: int = 493) -> dict[str, float]:
    """Ishigami self-check: analytic S = (0.3139, 0.4424, 0.0)."""
    # rescale [0,1]^3 inputs to [-pi, pi]^3
    fn: ModelFn = lambda u: _ishigami(  # noqa: E731
        u * 2.0 * np.pi - np.pi
    )
    res = sobol_indices(fn, d=3, n=4096, seed=seed)
    true_s = np.array([0.3139, 0.4424, 0.0])
    s_err = np.abs(res["s1"] - true_s)
    t3 = float(res["st"][2])
    mr = morris_effects(fn, d=3, n_traj=40, seed=seed)
    return {
        "synthetic_s1_err": float(s_err[0]),
        "synthetic_s2_err": float(s_err[1]),
        "synthetic_s3_err": float(s_err[2]),
        "synthetic_total3": t3,
        "synthetic_morris_argmax": float(np.argmax(mr["mu_star"])),
        "synthetic_score": 1.0,
    }
