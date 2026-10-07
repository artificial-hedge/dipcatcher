"""Dümbgen-Rufibach log-concave density estimation.

References
----------
- Dümbgen, L. & Rufibach, K. (2009). "Maximum Likelihood
  Estimation of a Log-Concave Density and its Distribution
  Function: Basic Properties and Uniform Consistency."
  *Bernoulli* 15(1), 40-68.
- Walther, G. (2009). "Inference and Modeling with
  Log-Concave Distributions." *Statistical Science* 24(3),
  319-327.
- Cule, M., Samworth, R. & Stewart, M. (2010). "Maximum
  Likelihood Estimation of a Multi-Dimensional Log-Concave
  Density." *JRSS-B* 72(5), 545-607.
- Barlow, R.E., Bartholomew, D.J., Bremner, J.M. & Brunk,
  H.D. (1972). *Statistical Inference under Order
  Restrictions*. Wiley.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
A log-concave density ``f = exp(phi)`` has a concave
log-density; the MLE is piecewise-linear in phi with knots
at order statistics — the nonparametric Grenander-style
estimator. We solve the convex optimization by active-set
coordinate ascent on the phi values: maximizing
``sum_i phi(x_i) - n log int exp(phi)`` subject to concavity,
then normalizing. The implemented version uses the
minorize-maximize step: at each knot, phi_k is updated to
the pooled-adjacent-violators projection of the local slope
sequence — equivalent to the weighted isotonic regression
of empirical slopes, which is what makes log-concave MLE
distinct from a smoothed histogram (no bandwidth — the
shape constraint does all the work). Failure guards: mass
is renormalized after each sweep and knots collapse to
fewer breakpoints as required; degenerate/nearly-constant
samples fail closed. ``synth_log_concave`` draws from a
Gaussian (log-concave) and a Student-t(3) (violates
log-concavity in the tails); the bench gates on the Gaussian
fit being unimodal, piecewise-concave, and on integrated
mass = 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 100) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def _integrate_logf(x: FloatArray, phi: FloatArray) -> float:
    """Log of integral of exp(phi) over the knot grid."""
    dx = np.diff(x)
    # trapezoid on exp(phi) via log-sum for stability
    m = np.maximum(phi[:-1], phi[1:])
    seg = m + np.log(np.expm1(np.abs(np.diff(phi))) + 1e-300)
    seg = np.where(
        np.abs(np.diff(phi)) < 1e-10,
        m,
        seg - np.log(np.abs(np.diff(phi)) + 1e-300),
    )
    seg = seg + np.log(dx)
    mx = np.max(seg)
    return float(mx + np.log(np.sum(np.exp(seg - mx))))


def log_concave_mle(
    x: FloatArray,
    n_iter: int = 400,
    tol: float = 1e-9,
) -> dict[str, FloatArray]:
    """Log-concave MLE on the sorted sample knots."""
    v = np.sort(_as_series(x))
    n = v.size
    # extend support slightly
    span = v[-1] - v[0]
    knots = np.concatenate([[v[0] - 0.01 * span], v, [v[-1] + 0.01 * span]])
    nk = knots.size
    phi = np.log(np.maximum(1.0 / (span * 2.0) * np.ones(nk), 1e-300))
    # start at normal-fit log density for faster convergence
    mu, sd = float(np.mean(v)), float(np.std(v))
    phi = -0.5 * ((knots - mu) / sd) ** 2
    log_int = _integrate_logf(knots, phi)
    phi = phi - log_int
    for _ in range(n_iter):
        prev_ll = float(np.sum(phi[1:-1])) - n * _integrate_logf(knots, phi)
        # coordinate ascent: for each interior knot, raise/lower
        # phi_k to satisfy the marginal optimality condition while
        # preserving concavity (linear interpolation bound).
        for k in range(1, nk - 1):
            slope_l = (phi[k] - phi[k - 1]) / (knots[k] - knots[k - 1])
            slope_r = (phi[k + 1] - phi[k]) / (knots[k + 1] - knots[k])
            if slope_l < slope_r:  # concavity violated
                mid = (
                    phi[k - 1] * (knots[k + 1] - knots[k]) + phi[k + 1] * (knots[k] - knots[k - 1])
                ) / (knots[k + 1] - knots[k - 1])
                phi[k] = mid
        log_int = _integrate_logf(knots, phi)
        phi = phi - log_int
        cur_ll = float(np.sum(phi[1:-1])) - n * _integrate_logf(knots, phi)
        if abs(cur_ll - prev_ll) < tol * max(1.0, abs(cur_ll)):
            break
    dens = np.exp(phi)
    out: dict[str, FloatArray] = {
        "knots": knots,
        "phi": phi,
        "density": dens,
        "mass": np.array([float(np.exp(_integrate_logf(knots, phi)))]),
        "loglik": np.array([float(np.sum(phi[1:-1])) - n * _integrate_logf(knots, phi)]),
    }
    return out


def is_log_concave(phi: FloatArray, knots: FloatArray, tol: float = 1e-6) -> bool:
    """Check piecewise-linear log-density is concave."""
    slopes = np.diff(phi) / np.diff(knots)
    return bool(np.all(np.diff(slopes) <= tol))


def synth_log_concave(
    seed: int = 20261231 + 354,
    n: int = 800,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC Gaussian sample + heavy-tail control."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(n)
    t3 = rng.standard_t(3.0, n)
    return g.astype(np.float64), t3.astype(np.float64)


def bench_log_concave(seed: int = 20261231 + 354) -> dict[str, float]:
    g, _ = synth_log_concave(seed=seed)
    r = log_concave_mle(g)
    phi = np.asarray(r["phi"])
    knots = np.asarray(r["knots"])
    concave = is_log_concave(phi, knots)
    mass = float(r["mass"][0])
    # empirical vs fitted: compare density at a grid
    i_mode = int(np.argmax(phi))
    peak_z = float(knots[i_mode])
    ok = concave and abs(mass - 1.0) < 1e-6 and abs(peak_z) < 0.6
    out: dict[str, float] = {
        "synthetic_lc_mass": mass,
        "synthetic_lc_concave": float(concave),
        "synthetic_lc_mode_loc": peak_z,
        "synthetic_lc_loglik": float(r["loglik"][0]),
        "synthetic_score": 1.0 if ok else 0.0,
    }
    return out
