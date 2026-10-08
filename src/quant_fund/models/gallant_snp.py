"""Gallant-Nychka semi-nonparametric (SNP) density estimation.

References
----------
- Gallant, A.R. & Nychka, D.W. (1987). "Semi-Nonparametric
  Maximum Likelihood Estimation." *Econometrica* 55(2), 363-390.
- Gallant, A.R. & Tauchen, G. (1989). "Seminonparametric
  Estimation of Conditionally Constrained Heterogeneous
  Processes: Asset Pricing Applications." *Econometrica* 57(5),
  1091-1120.
- Fenton, V.M. & Gallant, A.R. (1996). "Qualitative and
  Asymptotic Performance of SNP Density Estimators."
  *Journal of Econometrics* 74(1), 77-118.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
SNP multiplies a standard Gaussian kernel by the square of a
Hermite-polynomial expansion:
``f(z) = (sum_k a_k H_k(z))^2 * phi(z) / int (sum a_k H_k)^2 phi``
which nests the normal at K=0 and stays nonnegative by
construction — the squared-polynomial form (not signed) is what
makes SNP a proper density for every coefficient vector. We
standardize data first and estimate ``a_k`` by penalized ML
with a small quadratic penalty on coefficients — unpenalized
ML on a Hermite basis is unstable at moderate K (the leading
terms overfit tail bumps and the Hessian goes singular); the
``lam * a'a`` ridge keeps the optimizer on the interior.
``synth_snp`` draws from a Student-t with excess kurtosis —
Gaussian K=0 cannot capture it, K=4 can — plus a normal
control; the bench gates on the t-density log-likelihood gain
over Gaussian and on the normal fit preferring low K.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 200) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def _hermite_probabilists(z: FloatArray, k: int) -> FloatArray:
    """Probabilists' Hermite polynomials H_0..H_K evaluated."""
    out = np.zeros((z.size, k + 1))
    out[:, 0] = 1.0
    if k >= 1:
        out[:, 1] = z
    for j in range(2, k + 1):
        out[:, j] = z * out[:, j - 1] - (j - 1) * out[:, j - 2]
    return out


def snp_fit(
    x: FloatArray,
    k: int = 4,
    lam: float = 1e-3,
) -> dict[str, float]:
    """Penalized-ML SNP density fit on standardized data."""
    v = _as_series(x)
    if k < 0 or k > 8 or lam < 0:
        raise ValueError("bad expansion order")
    mu = float(np.mean(v))
    sd = float(np.std(v))
    z = (v - mu) / sd
    h = _hermite_probabilists(z, k)

    def nll(a: FloatArray) -> float:
        p2 = (h @ a) ** 2
        # normalizing constant via Gauss-Hermite quadrature
        w_pts, w_w = np.polynomial.hermite_e.hermegauss(48)
        hp = _hermite_probabilists(w_pts, k)
        norm_const = float(np.sum(w_w * (hp @ a) ** 2) / np.sqrt(2 * np.pi))
        if norm_const <= 0 or not np.isfinite(norm_const):
            return 1e10
        dens = p2 * norm.pdf(z) / norm_const
        ll = -float(np.sum(np.log(np.maximum(dens, 1e-300))))
        return ll + lam * float(np.sum(a[1:] ** 2))

    a0 = np.zeros(k + 1)
    a0[0] = 1.0
    res = minimize(nll, a0, method="BFGS", options={"maxiter": 800})
    ll_gauss = float(np.sum(norm.logpdf(z)))
    ll_snp = -float(res.fun) + lam * float(np.sum(res.x[1:] ** 2))
    out: dict[str, float] = {
        "ll_gauss": ll_gauss,
        "ll_snp": ll_snp,
        "ll_gain": ll_snp - ll_gauss,
        "converged": float(res.success),
        "k": float(k),
    }
    return out


def snp_density(
    x: FloatArray,
    grid: FloatArray,
    k: int = 4,
    lam: float = 1e-3,
) -> dict[str, FloatArray | float]:
    """Evaluate fitted SNP density on a grid."""
    v = _as_series(x)
    g = np.asarray(grid, dtype=np.float64)
    if g.size < 2:
        raise ValueError("bad grid")
    fit = snp_fit(v, k=k, lam=lam)
    mu = float(np.mean(v))
    sd = float(np.std(v))
    z = (v - mu) / sd
    zg = (g - mu) / sd
    h = _hermite_probabilists(z, k)
    hg = _hermite_probabilists(zg, k)

    # re-solve to get a (snp_fit doesn't return coefs — inline refit)
    def nll(a: FloatArray) -> float:
        w_pts, w_w = np.polynomial.hermite_e.hermegauss(48)
        hp = _hermite_probabilists(w_pts, k)
        nc = float(np.sum(w_w * (hp @ a) ** 2) / np.sqrt(2 * np.pi))
        p2 = (h @ a) ** 2
        dens = p2 * norm.pdf(z) / nc
        return -float(np.sum(np.log(np.maximum(dens, 1e-300)))) + lam * float(np.sum(a[1:] ** 2))

    a0 = np.zeros(k + 1)
    a0[0] = 1.0
    res = minimize(nll, a0, method="BFGS", options={"maxiter": 800})
    a = res.x
    w_pts, w_w = np.polynomial.hermite_e.hermegauss(48)
    hp = _hermite_probabilists(w_pts, k)
    nc = float(np.sum(w_w * (hp @ a) ** 2) / np.sqrt(2 * np.pi))
    dens_g = (hg @ a) ** 2 * norm.pdf(zg) / nc / sd
    return {
        "grid": g,
        "density": np.asarray(dens_g, dtype=np.float64),
        "ll_gain": float(fit["ll_gain"]),
    }


def synth_snp(
    seed: int = 20261231 + 343,
    n: int = 1500,
    df: float = 6.0,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC Student-t sample (heavy tails) vs normal."""
    rng = np.random.default_rng(seed)
    t_samp = rng.standard_t(df, size=n)
    n_samp = rng.standard_normal(n)
    return t_samp.astype(np.float64), n_samp.astype(np.float64)


def bench_gallant_snp(seed: int = 20261231 + 343) -> dict[str, float]:
    t_samp, n_samp = synth_snp(seed=seed)
    r_t = snp_fit(t_samp, k=4)
    r_n = snp_fit(n_samp, k=4)
    ok = r_t["ll_gain"] > 10.0 and abs(r_n["ll_gain"]) < 10.0
    out: dict[str, float] = {
        "synthetic_snp_ll_gain_t": r_t["ll_gain"],
        "synthetic_snp_ll_gain_normal": r_n["ll_gain"],
        "synthetic_snp_converged": r_t["converged"],
        "synthetic_score": 1.0 if ok else 0.0,
    }
    return out
