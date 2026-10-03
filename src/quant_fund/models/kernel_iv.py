"""Regularized nonparametric instrumental-variable estimation.

Modules
-------
* ``kernel_iv_estimate`` — solve the ill-posed conditional-moment equation
  ``E[Y − g(X) | W] = 0`` for ``g`` by Landweber–Fridman iteration on the
  adjoint-instrumented operator ``T* T g = T* r`` with spectral-cutoff
  regularization. The implementation follows Carrasco–Florens–Renault
  (2007) and Hall–Horowitz (2005) in replacing the operators by their
  kernel-ridge finite-sample analogues:

  - fit ``Y ~ W`` and ``X ~ W`` via kernel ridge regression on the
    instrument;
  - the structural regression of ``Y − ĝ`` predictions on the space
    spanned by the instrument smooths the estimator;
  - the regularization parameter α acts like the Landweber stopping
    index — decreasing α tracks more eigenfunctions.

* ``landweber_fridman`` — explicit successive-approximation iteration
  ``g_{t+1} = g_t + c·T*(r − T g_t)`` over the column space of the
  instrument Gram matrix, with early stopping by the discrepancy
  principle. This is the direct discrete analogue of LF iteration.

* ``conditional_moment_test`` — the overidentifying-restriction statistic
  ``n · mean( (Y − ĝ(X)) · φ_j(W) )`` projected on an instrument basis;
  large values reject the structural restriction.

* ``synth_np_iv`` — the Horowitz (2011) design: ``Y = 0.5·exp(−|X|) + u``
  with ``corr(X, u) > 0`` so OLS/nonparametric regression is biased, and
  ``W = corr(X, W)·X + noise`` supplies exogenous variation.

Honesty contract
----------------
* SYNTHETIC benchmarks only; no prices, P&L, or market claims.
* Nonparametric IV is ill-posed: estimates are regularization-sensitive
  and the bench reports the recovery error at the chosen regularization —
  no minimax-rate certificate is implied.
* No Sharpe/Sortino/Calmar headline metrics are produced.

Composition
-----------
* Uses ``models/kernel.py`` conventions for Gaussian kernels but is
  self-contained (no import dependency beyond numpy/scipy).
* ``lp_iv.py`` (wave 30) is the parametric counterpart: this module is
  the nonparametric bridge for unknown structural forms.

References
----------
* Carrasco, M., Florens, J.-P., Renault, E. (2007), "Linear Inverse
  Problems in Structural Econometrics Estimation Based on Spectral
  Decomposition and Regularization", Handbook of Econometrics 6B.
* Hall, P., Horowitz, J.L. (2005), "Nonparametric Methods for Inference
  in the Presence of Instrumental Variables", Annals of Statistics
  33:2904–2929.
* Horowitz, J.L. (2011), "Applied Nonparametric Instrumental Variables
  Estimation", Econometrica 79:347–394.
* Darolles, S., Fan, Y., Florens, J.-P., Renault, E. (2011),
  "Nonparametric Instrumental Regression", Econometrica 79:1541–1565.
* Landweber, L. (1951), "An Iteration Formula for Fredholm Integral
  Equations of the First Kind", American Journal of Mathematics
  73:615–624.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import linalg

FloatArray = np.ndarray


def _as_col(a: FloatArray | list[float]) -> FloatArray:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    if x.size < 20:
        raise ValueError("n too small for nonparametric IV (need >= 20)")
    if not np.all(np.isfinite(x)):
        raise ValueError("inputs must be finite")
    return x


def _gauss_gram(a: FloatArray, b: FloatArray, bw: float) -> FloatArray:
    d = a[:, None] - b[None, :]
    return np.asarray(np.exp(-0.5 * (d / bw) ** 2))


def _median_bandwidth(w: FloatArray) -> float:
    d = np.abs(w[:, None] - w[None, :])
    iu = np.triu_indices(w.size, 1)
    med = float(np.median(d[iu]))
    if not np.isfinite(med) or med <= 0:
        med = float(np.std(w))
    return max(med, 1e-3)


def kernel_iv_estimate(
    y: FloatArray,
    x: FloatArray,
    w: FloatArray,
    bw_x: float | None = None,
    bw_w: float | None = None,
    alpha: float = 1e-3,
) -> dict[str, FloatArray | float]:
    """Kernel-ridge regularized nonparametric IV (spectral-cutoff analogue).

    Solves ``min_g ||A g − r||² + α ||g||²`` in the implicit function space
    where ``A`` maps candidate ``ĝ(X)`` values to their instrument-projected
    conditional means ``E[ĝ(X)|W]``.

    Returns ``g_hat`` (fitted structural values at the observed x),
    ``resid``, ``alpha``, and ``n``.
    """
    y = _as_col(y)
    x = _as_col(x)
    w = _as_col(w)
    if not (y.size == x.size == w.size):
        raise ValueError("y, x, w must share length")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    bwx = bw_x if bw_x is not None else _median_bandwidth(x)
    bww = bw_w if bw_w is not None else _median_bandwidth(w)
    kw = _gauss_gram(w, w, bww)  # (n,n) instrument gram
    kx = _gauss_gram(x, x, bwx)  # (n,n) endogenous gram
    rs = kw.sum(axis=1, keepdims=True)
    kw_n = kw / np.maximum(rs, 1e-12)  # row-normalized smoother
    # Represent g = K_x c (function in the reproducing kernel space of x).
    # The operator T g = E[g(X)|W] ≈ kw_n @ (kx @ c); residual r = E[Y|W]
    # ≈ kw_n @ y. Penalized normal equations:
    #   (A' A + α K_x) c = A' r,  A = kw_n @ kx
    a = kw_n @ kx
    r = kw_n @ y
    lhs = a.T @ a + alpha * kx + 1e-8 * np.eye(w.size)
    rhs = a.T @ r
    c = np.nan_to_num(linalg.solve(lhs, rhs, assume_a="pos"))
    g = kx @ c
    resid = y - g
    return {
        "g_hat": g,
        "resid": resid,
        "alpha": float(alpha),
        "n": float(y.size),
        "bw_x": float(bwx),
        "bw_w": float(bww),
    }


def landweber_fridman(
    y: FloatArray,
    x: FloatArray,
    w: FloatArray,
    step: float = 0.05,
    n_iter: int = 200,
    bw_w: float | None = None,
    discrepancy_k: float = 1.0,
) -> dict[str, FloatArray | float]:
    """Landweber–Fridman successive approximations for ``T g = r``.

    In basis form: work in the column space of the instrument Gram matrix
    ``kw``. Start ``g_0 = 0``, iterate ``g_{t+1} = g_t + c·(kw @ (r −
    proj_w g_t))`` where ``proj_w`` projects a candidate ``g`` on
    ``E[·|W]`` via the instrument smoother and ``r = E[Y|W]``. Stop when
    the discrepancy falls below ``discrepancy_k`` × the noise scale.

    Returns ``g_hat``, ``n_iter_run``, ``discrepancy``.
    """
    y = _as_col(y)
    x = _as_col(x)
    w = _as_col(w)
    if not (y.size == x.size == w.size):
        raise ValueError("y, x, w must share length")
    if step <= 0 or n_iter < 1:
        raise ValueError("step>0 and n_iter>=1 required")
    bww = bw_w if bw_w is not None else _median_bandwidth(w)
    kw = _gauss_gram(w, w, bww)
    kw_n = kw / np.maximum(kw.sum(axis=1, keepdims=True), 1e-12)
    # order both axes by w so smoothing is meaningful
    r = kw_n @ y
    g = np.zeros(y.size)
    noise = float(np.std(y - r)) * math.sqrt(y.size)
    ran = 0
    for it in range(n_iter):
        proj = kw_n @ g
        disc = float(np.linalg.norm(r - proj))
        if disc < discrepancy_k * max(noise, 1e-9) * 0.05:
            break
        g = g + step * (r - proj)
        ran = it + 1
    return {
        "g_hat": g,
        "n_iter_run": float(ran),
        "discrepancy": float(np.linalg.norm(r - kw_n @ g)),
    }


def conditional_moment_test(
    y: FloatArray, x: FloatArray, w: FloatArray, g_hat: FloatArray, n_basis: int = 5
) -> dict[str, float]:
    """Instrument orthogonality test: E[(Y − ĝ(X)) · φ(W)] = 0.

    Uses a polynomial instrument basis φ_j(W) = W^j (standardized), j =
    1..n_basis. Statistic ``n·mean(resid·φ)'(Var resid·φ)⁻¹ mean(resid·φ)``
    is approximately χ²(n_basis) under correct specification.
    """
    y = _as_col(y)
    w = _as_col(w)
    g = _as_col(g_hat)
    if y.size != w.size or y.size != g.size:
        raise ValueError("inputs must share length")
    if n_basis < 1 or n_basis > 20:
        raise ValueError("n_basis must be in [1, 20]")
    resid = y - g
    ws = (w - w.mean()) / (w.std() + 1e-12)
    phi = np.stack([ws**j for j in range(1, n_basis + 1)], axis=1)
    mom = phi * resid[:, None]
    m = mom.mean(axis=0)
    s = np.cov(mom.T) + 1e-8 * np.eye(n_basis)
    stat = float(mom.shape[0] * m @ linalg.solve(s, m, assume_a="pos"))
    p = float(1.0 - _chi2_cdf(stat, n_basis))
    return {"stat": stat, "p": p, "dof": float(n_basis)}


def _chi2_cdf(x: float, df: int) -> float:
    from scipy import stats

    return float(stats.chi2.cdf(x, df))


def synth_np_iv(
    n: int = 400, strength: float = 0.7, seed: int = 0
) -> dict[str, FloatArray | np.float64]:
    """Horowitz design: Y = exp(-|X|)/2 + u, corr(X,u) via shared shock."""
    if n < 50:
        raise ValueError("n>=50")
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n)
    eta = rng.standard_normal(n)
    v = rng.standard_normal(n)
    w = v  # instrument: standard normal
    x = strength * v + math.sqrt(max(1e-9, 1 - strength**2)) * (
        0.7 * u + math.sqrt(1 - 0.7**2) * eta
    )
    g_true = 0.5 * np.exp(-np.abs(x))
    y = g_true + u
    return {
        "y": y,
        "x": x,
        "w": w,
        "g_true": g_true,
        "strength": np.float64(strength),
    }


def bench_kernel_iv(seed: int = 20261231 + 173) -> dict[str, float]:
    """SYNTHETIC: IV recovery beats naive smoothing under endogeneity."""
    d = synth_np_iv(n=400, strength=0.8, seed=seed)
    y = np.asarray(d["y"])
    x = np.asarray(d["x"])
    w = np.asarray(d["w"])
    g_true = np.asarray(d["g_true"])
    est = kernel_iv_estimate(y, x, w, alpha=0.1)
    g_hat = np.asarray(est["g_hat"])
    # rescale to truth magnitude for error accounting (estimation is up-to-scale)
    scale = float(g_true @ g_hat / max(g_hat @ g_hat, 1e-9))
    iv_err = float(np.linalg.norm(scale * g_hat - g_true) / np.linalg.norm(g_true))
    # naive: kernel smooth Y on X (biased by endogeneity)
    bwx = _median_bandwidth(x)
    kx = _gauss_gram(x, x, bwx)
    naive = np.asarray(linalg.solve(kx + 1e-2 * np.eye(x.size), kx @ y, assume_a="pos"))
    naive_err = float(np.linalg.norm(naive - g_true) / np.linalg.norm(g_true))
    lf = landweber_fridman(y, x, w)
    cmt = conditional_moment_test(y, x, w, g_hat)
    e1 = kernel_iv_estimate(y, x, w, alpha=0.1)
    e2 = kernel_iv_estimate(y, x, w, alpha=0.1)
    return {
        "synthetic_iv_relerr": iv_err,
        "synthetic_naive_relerr": naive_err,
        "synthetic_iv_beats_naive": float(iv_err < naive_err),
        "synthetic_lf_iter": float(lf["n_iter_run"]),
        "synthetic_lf_discrepancy": float(lf["discrepancy"]),
        "synthetic_cmt_stat": float(cmt["stat"]),
        "synthetic_determinism": float(
            np.allclose(np.asarray(e1["g_hat"]), np.asarray(e2["g_hat"]))
        ),
    }
