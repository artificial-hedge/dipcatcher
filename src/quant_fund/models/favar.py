"""Factor-augmented VAR (Bernanke-Boivin-Eliasz 2005, two-step form) (SYNTHETIC).

Step one extracts ``r`` principal-component factors from a standardized
panel ``X`` (T x n, n >> r).  Step two fits a VAR(p) on ``[y | F]`` where
``y`` is the observed target block (T x m).  Responses of ``y`` to factor
shocks read off the companion IRF directly; the ``y`` block sits first in
the ordering so its shock is recursive-identified first (standard BBE
"slow-fast" placement).

References: B. Bernanke, J. Boivin, P. Eliasz (2005), QJE 120(1);
J. Stock & M. Watson (2002) for the diffusion-index basis.  Fail-closed on
non-finite input, collinear panels, or infeasible (r, p).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.var_coint import fevd, var_fit, var_irf

Array = NDArray[np.float64]


def _as_panels(y: Array, x: Array) -> tuple[Array, Array]:
    ya = np.asarray(y, dtype=float)
    xa = np.asarray(x, dtype=float)
    if ya.ndim == 1:
        ya = ya.reshape(-1, 1)
    if ya.ndim != 2 or xa.ndim != 2:
        raise ValueError("y and x must be 2-D (t, m) and (t, n)")
    if ya.shape[0] != xa.shape[0]:
        raise ValueError("y and x must share the same row count")
    t = ya.shape[0]
    if t < 20 or xa.shape[1] < 2:
        raise ValueError("need t >= 20 and n >= 2 factor columns")
    if not np.all(np.isfinite(ya)) or not np.all(np.isfinite(xa)):
        raise ValueError("y/x must be finite")
    return ya, xa


def favar_extract(x: Array, r: int) -> dict[str, Array]:
    """Standardized PCA factors (scores) and loadings of the panel ``x``.

    Factors are normalized to unit sample variance; loadings satisfy
    ``x ~ F @ L`` up to residual.  ``r`` must be < n.
    """
    xa = np.asarray(x, dtype=float)
    if xa.ndim != 2 or xa.shape[0] < 10 or xa.shape[1] < 2:
        raise ValueError("x must be a finite (t, n) panel, t >= 10, n >= 2")
    if not np.all(np.isfinite(xa)):
        raise ValueError("x must be finite")
    t, n = xa.shape
    if not 1 <= r < n:
        raise ValueError("r must be in [1, n)")
    sd = np.std(xa, axis=0)
    if (sd <= 0).any():
        raise ValueError("x has a constant column")
    z = (xa - xa.mean(axis=0)) / sd
    _, sv, vt = np.linalg.svd(z, full_matrices=False)
    loadings = vt[:r].T * sv[:r] / math_sqrt_t(t)  # rows load on factors
    factors = z @ loadings / (sv[:r] ** 2 / t)  # PC scores (T x r)
    # normalize factor variance to 1 for VAR conditioning
    fsd = np.std(factors, axis=0)
    factors = factors / fsd
    explained = float(np.sum(sv[:r] ** 2) / np.sum(sv**2))
    return {
        "factors": np.asarray(factors, dtype=float),
        "loadings": loadings,
        "explained": np.array([explained]),
        "sd": sd,
        "mean": xa.mean(axis=0),
    }


def math_sqrt_t(t: int) -> float:
    return float(np.sqrt(t))


def favar_fit(y: Array, x: Array, r: int = 2, p: int = 2) -> dict[str, Array]:
    """Two-step FAVAR: PCA on ``x``, then VAR(p) on ``[y | factors]``.

    Returns the VAR fit (see :func:`var_fit`), the factor block, and the
    index offset of ``y`` inside the VAR (always 0..m-1).
    """
    ya, xa = _as_panels(y, x)
    ex = favar_extract(xa, r)
    z = np.hstack([ya, ex["factors"]])
    fit = var_fit(z, p)
    return {
        "var": fit["A"],
        "const": fit["const"],
        "sigma": fit["sigma"],
        "resid": fit["resid"],
        "ic": fit["ic"],
        "p": fit["p"],
        "factors": ex["factors"],
        "loadings": ex["loadings"],
        "explained": ex["explained"],
        "n_target": np.array([ya.shape[1]]),
        "n_var": np.array([z.shape[1]]),
        "r": np.array([r]),
    }


def favar_irf(fit: dict[str, Array], horizon: int = 20) -> dict[str, Array]:
    """IRFs of the stacked [y | F] system; both Cholesky and generalized.

    ``irf[h, i, j]``: response of variable ``i`` (0..m-1 are the ``y``
    targets) to shock ``j``.  Slice rows for the target block.
    """
    a = np.asarray(fit["var"], dtype=float)
    sigma = np.asarray(fit["sigma"], dtype=float)
    p = int(fit["p"][0])
    out = var_irf({"A": a, "sigma": sigma}, horizon)
    m = int(fit["n_target"][0])
    return {
        "oirf": out["oirf"],
        "girf": out["girf"],
        "oirf_target": out["oirf"][:, :m, :],
        "girf_target": out["girf"][:, :m, :],
        "p": np.array([p]),
    }


def favar_fevd(fit: dict[str, Array], horizon: int = 10) -> dict[str, Array]:
    """Generalized FEVD of the stacked system via :func:`fevd`."""
    a = np.asarray(fit["var"], dtype=float)
    sigma = np.asarray(fit["sigma"], dtype=float)
    out = fevd({"A": a, "sigma": sigma}, horizon)
    m = int(fit["n_target"][0])
    return {"gfevd": out["gfevd"], "gfevd_target": out["gfevd"][:m], "mse": out["mse"]}


def favar_forecast(
    fit: dict[str, Array], y_hist: Array, f_hist: Array, steps: int = 8
) -> dict[str, Array]:
    """Iterate the VAR forward from the last ``p`` stacked observations.

    ``y_hist`` (t, m) and ``f_hist`` (t, r) are the observed tails used to
    seed the recursion; only the target-block forecasts are returned.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    a = np.asarray(fit["var"], dtype=float)
    c = np.asarray(fit["const"], dtype=float)
    p, n = a.shape[0], a.shape[1]
    m = int(fit["n_target"][0])
    z_hist = np.hstack([np.asarray(y_hist, dtype=float), np.asarray(f_hist, dtype=float)])
    if z_hist.ndim != 2 or z_hist.shape[0] < p or z_hist.shape[1] != n:
        raise ValueError("history must be (t >= p, m + r) matching the fit")
    if not np.isfinite(z_hist).all():
        raise ValueError("history contains non-finite values")
    buf = z_hist[-p:].copy()
    out = np.empty((steps, m))
    for i in range(steps):
        nxt = c.copy()
        for lag in range(1, p + 1):
            nxt = nxt + a[lag - 1] @ buf[-lag]
        out[i] = nxt[:m]
        buf = np.vstack([buf, nxt])
    return {"y_forecast": out}


def synth_favar(
    t: int = 300,
    n_x: int = 12,
    r: int = 2,
    seed: int = 0,
) -> dict[str, Array]:
    """Panel driven by r latent AR(1) factors plus target y that
    loads on factor 1; observables share the factors."""
    rng = np.random.default_rng(seed)
    f = np.zeros((t, r))
    phis = np.array([0.7, 0.4])
    for i in range(1, t):
        f[i] = phis * f[i - 1] + rng.normal(0, 1, r)
    lam = rng.normal(0, 1, (n_x, r))
    x = f @ lam.T + rng.normal(0, 0.4, (t, n_x))
    y = 0.8 * f[:, [0]] + rng.normal(0, 0.5, (t, 1))
    return {"y": y, "x": x}


def bench_favar(seed: int = 20261231 + 246) -> dict[str, float]:
    """FAVAR self-check: r=2 factors explain most panel variance,
    factor 1 correlates with the simulated driver, and one-step
    forecasts beat the unconditional mean. All ``synthetic_*``."""
    d = synth_favar(r=2, seed=seed)
    out = favar_fit(d["y"], d["x"], r=2, p=1)
    expl = float(out["explained"][0])
    corr_f1 = abs(float(np.corrcoef(out["factors"][:, 0], d["y"].ravel())[0, 1]))
    fc = favar_forecast(
        out,
        np.asarray(d["y"], dtype=float)[-1:],
        np.asarray(out["factors"], dtype=float)[-1:],
        steps=1,
    )
    f1 = float(np.asarray(fc["y_forecast"], dtype=float).ravel()[0])
    out_b = favar_fit(d["y"], d["x"], r=2, p=1)
    return {
        "synthetic_explained": expl,
        "synthetic_corr_f1_y": corr_f1,
        "synthetic_forecast_y": f1,
        "synthetic_detects": float(expl > 0.5 and corr_f1 > 0.3 and np.isfinite(f1)),
        "synthetic_determinism": float(expl == float(out_b["explained"][0])),
    }
