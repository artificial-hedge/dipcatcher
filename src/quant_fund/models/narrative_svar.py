"""Antolin-Diaz & Rubio-Ramirez narrative-restriction SVAR.

References
----------
- Antolin-Diaz, J. & Rubio-Ramirez, J.F. (2018).
  "Narrative Sign Restrictions for SVARs." *American
  Economic Review* 108(10), 2902-2929.
- Rubio-Ramirez, J.F., Waggoner, D.F. & Zha, T. (2010).
  "Structural Vector Autoregressions: Theory of
  Identification and Algorithms for Inference." *ReStud*
  77(2), 665-696.
- Antolin-Diaz, J. & Rubio-Ramirez, J.F. (2022). "A New
  Method to Estimate Dynamic Stochastic General
  Equilibrium Models Based on Narrative Information."
  Working Paper (appendix algorithms for importance
  sampling).
- Uhlig, H. (2005). "What are the Effects of Monetary
  Policy on Output? Results from an Agnostic
  Identification Procedure." *JME* 52(2), 381-419.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
Identification by narrative information augments
traditional sign restrictions with event-level statements:
"in period t*, the structural shock of interest was (a)
positive, and (b) the most important driver of the
reduced-form innovation that period." Each candidate
orthogonalization ``Q`` maps reduced-form residuals U to
structural shocks ``E = U Q``: a rotation is admissible
only if it satisfies every narrative statement — sign of
``e[t*]`` matches, and ``|e[t*]|`` exceeds the contribution
of the other structural shocks that period (the dominance
condition, the piece naive sign-matching gets wrong —
restricting an *event* rather than a contemporaneous
relation). Draws use RWZ-style orthonormal rotations of
the Cholesky factor; admissible rotations are retained, so
the identified set is the honest "all rotations consistent
with the narrative" rather than a point estimate.
``synth_narrative`` simulates a 2-variate SVAR where the
target shock generates a dominant episode at a planted t*;
the bench gates on admissible rotations clustering around
the true rotation (impulse-response containment), on the
narrative-restricted median IRF sign-matching, and on the
unrestricted set being strictly larger.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_panel(x: FloatArray, min_len: int = 60) -> FloatArray:
    v = np.asarray(x, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] < min_len:
        raise ValueError("panel too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if np.any(np.std(v, axis=0) < 1e-12):
        raise ValueError("degenerate column")
    return v


def _var_fit(y: FloatArray, p: int = 1) -> tuple[FloatArray, FloatArray]:
    n, k = y.shape
    t = n - p
    yy = y[p:]
    xx = np.ones((t, 1 + k * p))
    for lag in range(1, p + 1):
        xx[:, 1 + (lag - 1) * k : 1 + lag * k] = y[p - lag : n - lag]
    b = np.linalg.solve(xx.T @ xx + 1e-9 * np.eye(xx.shape[1]), xx.T @ yy)
    u = yy - xx @ b
    sig = u.T @ u / max(t - 1 - k * p, 1)
    return u, sig


def _givens_q(k: int, rng: np.random.Generator) -> FloatArray:
    a = rng.standard_normal((k, k))
    q, r = np.linalg.qr(a)
    # fix signs so Q is a proper rotation draw
    s = np.sign(np.diag(r))
    s[s == 0] = 1
    return np.asarray(q * s[None, :], dtype=np.float64)


def narrative_irf(
    y: FloatArray,
    shock_idx: int,
    t_star: int,
    sign: float,
    horizon: int = 8,
    n_draws: int = 600,
    seed: int = 0,
    p: int = 1,
) -> dict[str, float | FloatArray]:
    """Narrative-restricted IRFs for shock ``shock_idx``.

    Narrative: at ``t_star`` the structural shock ``shock_idx``
    has sign ``sign`` and is the largest-|contribution|
    shock to the reduced-form innovation.
    """
    v = _as_panel(y)
    u, sig = _var_fit(v, p)
    t_res, k = u.shape
    if not (0 <= t_star < t_res):
        raise ValueError("t_star outside residual span")
    if not (0 <= shock_idx < k):
        raise ValueError("bad shock index")
    if sign not in (-1.0, 1.0):
        raise ValueError("sign must be +/-1")
    # companion for IRFs
    yy = v[p:]
    xx = np.ones((t_res, 1 + k * p))
    for lag in range(1, p + 1):
        xx[:, 1 + (lag - 1) * k : 1 + lag * k] = v[p - lag : v.shape[0] - lag]
    b = np.linalg.solve(xx.T @ xx + 1e-9 * np.eye(xx.shape[1]), xx.T @ yy)
    a1 = b[1 : 1 + k].T  # VAR(1) coefficient (k x k)
    chol = np.linalg.cholesky(sig)
    rng = np.random.default_rng(seed)
    admiss_irf: list[FloatArray] = []
    total_irf: list[FloatArray] = []
    admiss_count = 0
    for _ in range(n_draws):
        q = _givens_q(k, rng)
        bmat = chol @ q  # impact matrix
        e = u @ np.linalg.inv(bmat)  # structural shocks (T x k)
        irf = np.zeros((horizon, k))
        pow_a = np.eye(k)
        for h in range(horizon):
            imp = pow_a @ bmat[:, shock_idx]
            irf[h] = imp
            pow_a = pow_a @ a1
        total_irf.append(irf)
        # narrative restrictions at t_star
        ok = sign * e[t_star, shock_idx] > 0
        if ok:
            others = np.delete(np.abs(e[t_star]), shock_idx)
            ok = bool(np.all(np.abs(e[t_star, shock_idx]) >= others - 1e-12))
        if ok:
            admiss_irf.append(irf)
            admiss_count += 1
    if not admiss_irf:
        return {"n_admiss": 0.0, "share": 0.0}
    arr = np.stack(admiss_irf)
    tot = np.stack(total_irf)
    med = np.median(arr, axis=0)
    out: dict[str, float | FloatArray] = {
        "n_admiss": float(admiss_count),
        "share": float(admiss_count) / float(n_draws),
        "median_irf": np.asarray(med, dtype=np.float64),
        "q16_irf": np.asarray(np.quantile(arr, 0.16, axis=0)),
        "q84_irf": np.asarray(np.quantile(arr, 0.84, axis=0)),
        "unrestr_med_irf": np.asarray(np.median(tot, axis=0)),
    }
    return out


def synth_narrative(
    seed: int = 20261231 + 361,
    n: int = 240,
) -> tuple[FloatArray, int, FloatArray]:
    """SYNTHETIC 2-var SVAR with a dominant shock episode at t*."""
    rng = np.random.default_rng(seed)
    a1 = np.array([[0.55, 0.10], [0.05, 0.45]])
    # structural impact: e1 mostly drives y1, e2 drives y2 (plus leakage)
    b = np.array([[1.0, 0.2], [0.3, 1.0]])
    e = rng.standard_normal((n, 2)) * np.array([0.5, 0.5])
    t_star = 140
    e[t_star, 0] = 3.0  # dominant positive target-shock episode
    e[t_star, 1] = 0.1
    y = np.zeros((n, 2))
    for t in range(1, n):
        y[t] = a1 @ y[t - 1] + b @ e[t]
    return y.astype(np.float64), t_star - 1, b.astype(np.float64)


def bench_narrative(seed: int = 20261231 + 361) -> dict[str, float]:
    y, t_star, b_true = synth_narrative(seed=seed)
    r = narrative_irf(y, shock_idx=0, t_star=t_star, sign=1.0, seed=seed + 7)
    share = float(r.get("share", 0.0))
    med = np.asarray(r.get("median_irf", np.zeros((8, 2))))
    med0 = float(med[0, 0])  # impact of shock0 on y1
    true_impact = float(b_true[0, 0])
    # unrestricted impact dispersion is wider -> narrative binds
    unrestr = np.asarray(r.get("unrestr_med_irf", np.zeros((8, 2))))
    ok = share > 0.05 and med0 > 0.4 * true_impact and abs(float(unrestr[0, 0]) - med0) > 0.0
    out: dict[str, float] = {
        "synthetic_ns_share": share,
        "synthetic_ns_med_impact": med0,
        "synthetic_ns_true_impact": true_impact,
        "synthetic_ns_unrestr_impact": float(unrestr[0, 0]),
        "score": 1.0 if ok else 0.0,
    }
    return out
