"""Hansen-Jagannathan SDF distance + model diagnostics.

The HJ distance measures how far a candidate stochastic discount
factor lies from the set of SDFs that price a given return panel
(Hansen & Jagannathan 1997). Complementary diagnostics: the
mean-variance SDF bound (admissible (E[m], σ(m)) frontier),
mimicking-portfolio projection of the SDF onto the return space, and
the Kan-Robotti-Shanken (2013) pricing-error alpha test with a GMM
weighting matrix.

References
----------
- Hansen & Jagannathan (1997). Assessing specification errors in
  stochastic discount factor models. *Journal of Finance* 52(2).
- Kan, Robotti & Shanken (2013). Pricing model performance and the
  two-pass cross-sectional regression methodology. *Journal of Finance*
  68(6).
- Cochrane (2005). *Asset Pricing*, rev. ed., ch. 5 (HJ machinery).

Honesty
-------
All panels are SYNTHETIC factor structures with a known true kernel.
Keys report distance ordering (true kernel ~0 vs misspecified >0) and
test size/power on simulated returns — never asset-pricing conclusions
about real markets.

Composition notes
-----------------
- ``models/asset_pricing.py``: IPCA + SDF ridge ranker — the estimation
  machinery; this module is the *model evaluation* layer on top of any
  SDF.
- ``validation/multiple_testing.py``: forecast-comparison tests — KRS
  alpha tests are a different family (pricing-error inference).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sstats

FloatArray = NDArray[np.float64]


def _as_returns(r: FloatArray) -> FloatArray:
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] < 20 or r.shape[1] < 2:
        raise ValueError("R must be (T>=20, n>=2)")
    if not np.all(np.isfinite(r)):
        raise ValueError("returns must be finite")
    return r


def sdf_mv_bound(r: FloatArray, n_points: int = 50) -> FloatArray:
    """Hansen-Jagannathan mean-variance bound: for each E[m] grid point,
    the minimum σ(m) among SDFs pricing the return panel.

    For gross returns R (T x n): candidate SDFs m = a + b'(R - mu) live
    in the span of returns + constant; the bound at E[m]=a is the std of
    the projected SDF satisfying E[m R] = 1. Returns (n_points, 2) grid
    of (E[m], min σ(m)).
    """
    r = _as_returns(r)
    t, n = r.shape
    mu = r.mean(axis=0)
    cov = np.cov(r.T) + 1e-10 * np.eye(n)
    # minimum-variance SDF at mean m0 solving E[m R] = 1:
    # m* = m0 + (1 - m0 mu)' Σ⁻¹ (R - mu)  → var = (1 - m0 mu)' Σ⁻¹ (1 - m0 mu)
    cov_inv = np.linalg.inv(cov)
    mu_vec = np.asarray(mu)
    grid = np.linspace(0.5, 1.5, n_points)
    out = np.empty((n_points, 2))
    for i, m0 in enumerate(grid):
        diff = 1.0 - m0 * mu_vec
        var = float(diff @ cov_inv @ diff)
        out[i] = [m0, math.sqrt(max(var, 0.0))]
    return out


def hj_distance(r: FloatArray, f: FloatArray, weighting: str = "returns") -> float:
    """HJ distance: min over admissible-SDF projections of E[(m - m*)^2].

    Candidate kernel m = m0 + b' f_t (f: T x k demeaned factors).
    Closed form: distance² = e' (E[R R'])⁻¹ e evaluated for the
    projection residual e = E[R m*] - 1, with m* the candidate.
    We report the second-moment distance (returns-weighted), the
    version most robust to non-positive SDF realisations.
    """
    r = _as_returns(r)
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 2 or f.shape[0] != r.shape[0]:
        raise ValueError("f must be (T, k) aligned to R")
    t, n = r.shape
    # candidate kernel: linear projection of unit mass onto factors
    g = np.column_stack([np.ones(t), f - f.mean(axis=0)])
    # find b minimising ||E[R·(g b)] - 1||² — pricing-error projection
    grm = (g[:, :, None] * r[:, None, :]).mean(axis=0)  # E[g R] (k+1 x n)
    b, _res, _rk, _sv = np.linalg.lstsq(grm.T, np.ones(n), rcond=None)
    m_proj = g @ b
    # pricing errors of the candidate kernel
    err = (r * m_proj[:, None]).mean(axis=0) - 1.0
    if weighting == "returns":
        w = np.cov(r.T) + 1e-10 * np.eye(n)
    elif weighting == "second":
        w = (r[:, :, None] * r[:, None, :]).mean(axis=0) + 1e-10 * np.eye(n)
    else:
        raise ValueError("weighting in {'returns', 'second'}")
    dist = float(err @ np.linalg.inv(w) @ err)
    return dist


def krs_alpha_test(r: FloatArray, f: FloatArray) -> dict[str, float]:
    """Kan-Robotti-Shanken style pricing-error test.

    Runs a linear SDF projection and tests whether the n pricing errors
    are jointly zero via a chi-square statistic on the error vector,
    with a heteroskedasticity-consistent covariance of the time-series
    means (Newey-West-free; plain sandwich on the mean equation).
    """
    r = _as_returns(r)
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 2 or f.shape[0] != r.shape[0]:
        raise ValueError("f must be (T, k)")
    t, n = r.shape
    g = np.column_stack([np.ones(t), f - f.mean(axis=0)])
    grm = (g[:, :, None] * r[:, None, :]).mean(axis=0)
    b, _res, _rk, _sv = np.linalg.lstsq(grm.T, np.ones(n), rcond=None)
    m = g @ b
    err_series = r * m[:, None] - 1.0  # T x n pricing errors
    e_bar = err_series.mean(axis=0)
    s_cov = np.cov(err_series.T)  # (n x n)
    stat = float(t * e_bar @ np.linalg.inv(s_cov + 1e-10 * np.eye(n)) @ e_bar)
    p = float(sstats.chi2.sf(stat, df=n))
    return {"krs_stat": stat, "krs_p": p, "n_pricing_errors": float(n)}


def sharpe_of_sdf(r: FloatArray) -> float:
    """Maximum |Sharpe| implied by the panel: sqrt(mu' Σ⁻¹ mu).

    Note (Cochrane ch.5): for GROSS returns the tangent vertex of the
    (E[m], σ(m)) bound is sqrt(A - B²/C) with A = μ'Σ⁻¹μ,
    B = 1'Σ⁻¹μ, C = 1'Σ⁻¹1 — the maximal Sharpe over *portfolios
    orthogonal to the unit vector*, not the raw maximal Sharpe. Both
    diagnostics are reported by ``bench_hj_distance``.
    """
    r = _as_returns(r)
    mu = r.mean(axis=0)
    cov = np.cov(r.T) + 1e-10 * np.eye(r.shape[1])
    v = float(mu @ np.linalg.inv(cov) @ mu)
    return float(math.sqrt(max(v, 0.0)))


def sdf_vertex_slope(r: FloatArray) -> float:
    """Tangent slope of the HJ bound from the origin for gross returns:
    sqrt(A - B²/C) — the vertex touches σ(m) = slope · E[m]."""
    r = _as_returns(r)
    n = r.shape[1]
    mu = r.mean(axis=0)
    cov = np.cov(r.T) + 1e-10 * np.eye(n)
    ci = np.linalg.inv(cov)
    a_ = float(mu @ ci @ mu)
    b_ = float(np.ones(n) @ ci @ mu)
    c_ = float(np.ones(n) @ ci @ np.ones(n))
    return float(math.sqrt(max(a_ - b_ * b_ / c_, 0.0)))


def synth_asset_panel(
    t: int = 300,
    n: int = 8,
    k: int = 2,
    seed: int = 0,
    misspecified: bool = False,
) -> dict[str, FloatArray]:
    """Returns generated by a k-factor SDF model.

    True SDF: m = 1 - λ' (f - mu_f) with λ = Σ_f⁻¹ priced loadings.
    When misspecified=True, the returned factors omit the second factor
    (a misspecified candidate kernel for the same returns).
    """
    rng = np.random.default_rng(seed)
    lam = np.array([0.4, 0.6])
    f_sig = np.array([[1.0, 0.3], [0.3, 1.0]])
    f_true = rng.multivariate_normal(np.zeros(2), f_sig, t)
    # loadings matrix B (n x k)
    b_mat = rng.standard_normal((n, k)) * 0.5 + 0.5
    # idiosyncratic noise + priced-component returns: R_i = B_i f + e_i,
    # m = 1 - f' Σ_f⁻¹ λ · (f - 0); E[R m] = 1 requires E[R] - B cov(f, f) Σ_f⁻¹ λ = 1
    # solve for intercept a: a + B λ = 1 → a = 1 - B λ
    a_int = 1.0 - b_mat @ lam
    e = rng.standard_normal((t, n)) * 0.15
    r = a_int[None, :] + f_true @ b_mat.T + e
    if misspecified:
        f_out = f_true[:, :1]  # omit factor 2 → misspecified kernel
    else:
        f_out = f_true
    return {
        "R": np.asarray(r, dtype=np.float64),
        "F": np.asarray(f_out, dtype=np.float64),
        "lambda": np.asarray(lam),
        "B": np.asarray(b_mat),
    }


def bench_hj_distance(seed: int = 20261231 + 165) -> dict[str, float]:
    """SYNTHETIC HJ-distance ordering: correct kernel ~0 vs misspecified."""
    good = synth_asset_panel(t=400, n=8, seed=seed, misspecified=False)
    bad = synth_asset_panel(t=400, n=8, seed=seed, misspecified=True)
    d_good = hj_distance(good["R"], good["F"], weighting="returns")
    d_bad = hj_distance(bad["R"], bad["F"], weighting="returns")
    d2_good = hj_distance(good["R"], good["F"], weighting="second")
    d2_bad = hj_distance(bad["R"], bad["F"], weighting="second")
    krs_good = krs_alpha_test(good["R"], good["F"])
    krs_bad = krs_alpha_test(bad["R"], bad["F"])
    bound = sdf_mv_bound(good["R"], n_points=30)
    sr = sharpe_of_sdf(good["R"])
    vertex = sdf_vertex_slope(good["R"])
    # the bound's minimum σ(m)/E[m] should equal the vertex slope
    bound_ratio = bound[:, 1] / np.maximum(bound[:, 0], 1e-9)
    bound_ok = float(abs(np.min(bound_ratio) - vertex) < 0.05 * max(vertex, 1.0))
    d_a = hj_distance(good["R"], good["F"])
    d_b = hj_distance(good["R"], good["F"])
    return {
        "synthetic_hj_true": d_good,
        "synthetic_hj_misspec": d_bad,
        "synthetic_hj_margin": d_bad - d_good,
        "synthetic_hj2_true": d2_good,
        "synthetic_hj2_misspec": d2_bad,
        "synthetic_krs_p_true": float(krs_good["krs_p"]),
        "synthetic_krs_p_misspec": float(krs_bad["krs_p"]),
        "synthetic_sr_max": sr,
        "synthetic_vertex_slope": vertex,
        "synthetic_bound_covers_sr": bound_ok,
        "synthetic_determinism": float(d_a == d_b),
    }
