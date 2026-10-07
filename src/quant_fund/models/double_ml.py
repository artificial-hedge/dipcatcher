"""Double/debiased machine learning (Chernozhukov et al. 2018).

Orthogonalized estimation of structural parameters when high-dimensional
nuisances ``g(X), m(X)`` are learned flexibly. Two canonical scores:

1. **Partially-linear regression (PLR)**: ``Y = θD + g(X) + U``,
   ``D = m(X) + V`` — the Neyman-orthogonal partialling-out score
   ``ψ = (Y − ĝ − θ(D − m̂))(D − m̂)``.

2. **Interactive regression (IRM / ATE)**: potential means
   ``g_d(X) = E[Y|D=d, X]`` and propensity ``m(X) = P(D=1|X)`` with the
   doubly-robust AIPW score.

Both use ``K``-fold **cross-fitting** (DML2: pooled score across folds)
so nuisance overfitting does not contaminate the parameter estimate.
Nuisance learners are kernel ridge regression on a Gaussian kernel
(bounded, deterministic) or ridge polynomial features for speed.

References
----------
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen,
  C., Newey, W. & Robins, J. (2018). *Double/debiased machine learning
  for treatment and structural parameters.* Econometrics Journal
  21(1), C1–C68.
- Robinson, P. M. (1988). *Root-n-consistent semiparametric
  regression.* Econometrica 56(4), 931–954.
- Athey, S. & Imbens, G. (2016). *Recursive partitioning for
  heterogeneous causal effects.* PNAS 113(27) — cross-fitting
  motivation.

Honesty contract
----------------
``synth_*`` helpers and ``bench_*`` emit SYNTHETIC correctness checks
only — never market evidence.

Composition notes
-----------------
numpy/scipy only; deterministic ``np.random.default_rng(seed)``;
fail-closed ``ValueError`` on weak residual variation (Riesz
representer degenerates); public dict keys never contain forbidden
score tokens.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import linalg

FloatArray = np.ndarray

__all__ = [
    "bench_double_ml",
    "dml_irm",
    "dml_plr",
    "synth_irm",
    "synth_plr",
]


def _gauss_kernel(x1: FloatArray, x2: FloatArray, bw: float) -> FloatArray:
    d = x1[:, None, :] - x2[None, :, :]
    return np.exp(-0.5 * np.sum(d * d, axis=2) / (bw * bw))


def _median_bw(x: FloatArray) -> float:
    i = np.arange(x.shape[0])
    d2 = np.sum((x[i[:, None] % x.shape[0]] - x[None, :]) ** 2, axis=2)
    med = float(np.median(np.sqrt(d2[d2 > 0])))
    return max(med, 1e-6)


def _krr_fit_predict(
    x_tr: FloatArray, y_tr: FloatArray, x_te: FloatArray, lam: float = 1.0
) -> FloatArray:
    """Gaussian-kernel ridge regression (bounded influence, deterministic)."""
    bw = _median_bw(x_tr)
    k_tr = _gauss_kernel(x_tr, x_tr, bw)
    k_te = _gauss_kernel(x_te, x_tr, bw)
    a = linalg.solve(k_tr + lam * np.eye(x_tr.shape[0]), y_tr, assume_a="pos")
    return np.asarray(k_te @ a)


def _poly_features(x: FloatArray, deg: int = 3) -> FloatArray:
    n, p = x.shape
    cols = [np.ones(n)]
    for d in range(1, deg + 1):
        for j in range(p):
            cols.append(x[:, j] ** d)
    for j in range(p):
        for k in range(j + 1, p):
            cols.append(x[:, j] * x[:, k])
    return np.column_stack(cols)


def _ridge_fit_predict(
    x_tr: FloatArray, y_tr: FloatArray, x_te: FloatArray, lam: float = 1e-3
) -> FloatArray:
    f_tr = _poly_features(x_tr)
    f_te = _poly_features(x_te)
    a = f_tr.T @ f_tr + lam * np.eye(f_tr.shape[1])
    b = linalg.solve(a, f_tr.T @ y_tr, assume_a="pos")
    return np.asarray(f_te @ b)


def _folds(n: int, k: int, rng: np.random.Generator) -> list[FloatArray]:
    idx = rng.permutation(n)
    return np.array_split(idx, k)


def _check_xyd(
    y: FloatArray, d: FloatArray, x: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    y = np.asarray(y, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim != 1 or d.shape != y.shape or x.shape[0] != y.size:
        raise ValueError("y, d same-length; x same rows")
    if y.size < 60:
        raise ValueError("need at least 60 observations")
    if x.shape[1] < 1 or x.shape[1] > 30:
        raise ValueError("x must have 1..30 columns")
    return y, d, x


def dml_plr(
    y: FloatArray,
    d: FloatArray,
    x: FloatArray,
    n_folds: int = 5,
    learner: str = "krr",
    seed: int = 0,
) -> dict[str, float]:
    """DML2 estimate of θ in ``Y = θD + g(X) + U``, ``D = m(X) + V``.

    Cross-fits ĝ and m̂ over ``n_folds`` folds, then
    ``θ = Σ (D−m̂)(Y−ĝ) / Σ (D−m̂)²`` with the orthogonal sandwich SE
    ``se² = Σ ψ² / (Σ (D−m̂)²)² · n`` scaled by 1/n via the usual form
    ``J = mean((D−m̂)²)``, ``σ² = mean(ψ²)/J²``.
    """
    y, d, x = _check_xyd(y, d, x)
    if not 2 <= n_folds <= 10:
        raise ValueError("n_folds in 2..10")
    rng = np.random.default_rng(seed)
    folds = _folds(y.size, n_folds, rng)
    predict = _krr_fit_predict if learner == "krr" else _ridge_fit_predict
    y_hat = np.empty_like(y)
    d_hat = np.empty_like(d)
    for te in folds:
        tr = np.setdiff1d(np.arange(y.size), te)
        y_hat[te] = predict(x[tr], y[tr], x[te])
        d_hat[te] = predict(x[tr], d[tr], x[te])
    v = d - d_hat
    u = y - y_hat
    j = float(np.mean(v * v))
    scale = max(float(np.mean(d * d)), 1e-12)
    if j < 1e-2 * scale:
        raise ValueError("D is (near-)perfectly predicted by X — Riesz representer degenerate")
    theta = float(np.sum(v * u) / np.sum(v * v))
    psi = (u - theta * v) * v
    se = math.sqrt(float(np.mean(psi * psi)) / (j * j) / y.size)
    return {
        "theta": theta,
        "se": se,
        "t": theta / max(se, 1e-12),
        "j": j,
        "nuisance_r2_y": float(1.0 - np.var(u) / max(np.var(y), 1e-12)),
        "nuisance_r2_d": float(1.0 - np.var(v) / max(np.var(d), 1e-12)),
    }


def dml_irm(
    y: FloatArray,
    d: FloatArray,
    x: FloatArray,
    n_folds: int = 5,
    learner: str = "krr",
    seed: int = 0,
) -> dict[str, float]:
    """Doubly-robust ATE via cross-fitted IRM scores.

    ``ψ = g1(X) − g0(X) + D(Y−g1)/m − (1−D)(Y−g0)/(1−m) − θ``;
    ``ATE = mean ψ + θ``. Propensity clipped to ``[0.01, 0.99]``.
    """
    y, d, x = _check_xyd(y, d, x)
    # Compare raw values, not truncations: astype(int) would silently map a
    # fractional dose like {0.3, 1.7} onto {0, 1} and the AIPW score would
    # then compute confidently wrong math on a non-binary treatment.
    if set(np.unique(d)) != {0, 1}:
        raise ValueError("d must be binary for IRM")
    rng = np.random.default_rng(seed)
    folds = _folds(y.size, n_folds, rng)
    predict = _krr_fit_predict if learner == "krr" else _ridge_fit_predict
    g0 = np.empty_like(y)
    g1 = np.empty_like(y)
    m = np.empty_like(y)
    for te in folds:
        tr = np.setdiff1d(np.arange(y.size), te)
        tr0 = tr[d[tr] <= 0.5]
        tr1 = tr[d[tr] > 0.5]
        g0[te] = predict(x[tr0], y[tr0], x[te])
        g1[te] = predict(x[tr1], y[tr1], x[te])
        m[te] = predict(x[tr], d[tr], x[te])
    m = np.clip(m, 0.01, 0.99)
    psi = g1 - g0 + d * (y - g1) / m - (1.0 - d) * (y - g0) / (1.0 - m)
    ate = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / math.sqrt(y.size))
    return {
        "ate": ate,
        "se": se,
        "t": ate / max(se, 1e-12),
        "mean_g1": float(np.mean(g1)),
        "mean_g0": float(np.mean(g0)),
        "propensity_min": float(m.min()),
        "propensity_max": float(m.max()),
    }


def synth_plr(
    n: int = 900,
    theta: float = 0.8,
    p: int = 4,
    seed: int = 0,
) -> dict[str, FloatArray | np.float64]:
    """SYNTHETIC PLR with nonlinear confounding:
    ``g(x) = sin(x1) + 0.5 x2²``, ``m(x) = 0.3 x1 + tanh(x3)`` so naive
    OLS on ``Y ~ D`` is badly biased."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, p))
    g = np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2
    m = 0.3 * x[:, 0] + np.tanh(x[:, 2])
    d = m + rng.normal(0, 0.7, n)
    y = theta * d + g + rng.normal(0, 0.6, n)
    return {"y": y, "d": d, "x": x, "theta_true": np.float64(theta)}


def synth_irm(
    n: int = 900,
    tau: float = 1.2,
    p: int = 4,
    seed: int = 0,
) -> dict[str, FloatArray | np.float64]:
    """SYNTHETIC IRM with confounded binary treatment:
    propensity ``σ(0.6x1 − 0.4x2)``, ``Y(0)=x·η + ε``,
    ``Y(1) = Y(0) + tau + 0.4·x3²``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, p))
    pr = 1.0 / (1.0 + np.exp(-(0.6 * x[:, 0] - 0.4 * x[:, 1])))
    d = (rng.random(n) < pr).astype(np.float64)
    y0 = 0.5 * x[:, 0] + 0.8 * np.sin(x[:, 1]) + rng.normal(0, 0.5, n)
    y = y0 + d * (tau + 0.4 * x[:, 2] ** 2)
    # E[x3^2] = 1 so the population ATE is tau + 0.4, not tau.
    return {"y": y, "d": d, "x": x, "tau_true": np.float64(tau + 0.4)}


def bench_double_ml(seed: int = 20261231 + 180) -> dict[str, float]:
    """SYNTHETIC: DML-PLR beats naive OLS; IRM ATE in the right band."""
    d = synth_plr(n=800, theta=0.8, seed=seed)
    y, t, x = np.asarray(d["y"]), np.asarray(d["d"]), np.asarray(d["x"])
    est = dml_plr(y, t, x, n_folds=4, learner="ridge", seed=seed)
    # naive OLS of y on [1, d]
    xd = np.column_stack([np.ones(y.size), t])
    b_ols = float(np.linalg.lstsq(xd, y, rcond=None)[0][1])
    di = synth_irm(n=800, tau=1.2, seed=seed + 1)
    ate = dml_irm(
        np.asarray(di["y"]),
        np.asarray(di["d"]),
        np.asarray(di["x"]),
        n_folds=4,
        learner="ridge",
        seed=seed + 1,
    )
    est2 = dml_plr(y, t, x, n_folds=4, learner="ridge", seed=seed)
    return {
        "synthetic_theta_hat": float(est["theta"]),
        "synthetic_theta_err": abs(float(est["theta"]) - 0.8),
        "synthetic_ols_err": abs(b_ols - 0.8),
        "synthetic_theta_t": float(est["t"]),
        "synthetic_ate_hat": float(ate["ate"]),
        "synthetic_ate_err": abs(float(ate["ate"]) - 1.6),
        "synthetic_determinism": float(est["theta"] == est2["theta"]),
    }
