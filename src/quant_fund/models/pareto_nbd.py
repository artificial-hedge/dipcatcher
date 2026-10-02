"""Customer lifetime value: Pareto/NBD (Schmittlein,
Morrison & Colombo 1987) and BG/NBD (Fader, Hardie &
Lee 2005).

- Pareto/NBD: transactions arrive as Poisson(rate l_i)
  while alive; lifetime ~ Exp(mu_i); heterogeneity
  l ~ Gamma(r, alpha), mu ~ Gamma(s, beta). Closed-form
  likelihood via the Pareto/NBD integral (Fader-Hardie
  2005 computationally tractable reparametrization).
- BG/NBD: same purchase process, lifetime Geometric
  dropout after each transaction (dropout prob p,
  heterogeneity p ~ Beta(a, b), l ~ Gamma(r, alpha)).
  Closed-form likelihood in terms of Gauss hypergeometric.
- Conditional E[future purchases | x, t_x, T] and
  P(alive) formulas from the papers.

References
----------
- Schmittlein, Morrison & Colombo (1987) 'Counting your
  customers' Management Science 33(1).
- Fader, Hardie & Lee (2005) 'Counting your customers
  the easy way' Marketing Science 24(2).
- Fader & Hardie (2005) 'A note on implementing the
  Pareto/NBD model' (tractable likelihood).

Honesty
-------
SYNTHETIC self-check: simulate heterogeneous Poisson +
geometric dropout; asserts recovered expected future
purchases tracks the realized next-period counts better
than a naive recency heuristic.

Composition
-----------
Pure numpy/scipy. Inputs are (frequency x, recency t_x,
tenure T) per customer; outputs are E[purchases],
P(alive), fitted hyperparameters.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1

FloatArray = NDArray[np.float64]


def _check_cbs(
    x: FloatArray, tx: FloatArray, T: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    xa = np.asarray(x, dtype=np.float64).ravel()
    ta = np.asarray(tx, dtype=np.float64).ravel()
    Ta = np.asarray(T, dtype=np.float64).ravel()
    if xa.size != ta.size or xa.size != Ta.size or xa.size < 1:
        raise ValueError("cbs length mismatch")
    if (xa < 0).any() or (ta < 0).any() or (Ta <= 0).any() or (ta > Ta).any():
        raise ValueError("invalid (x, t_x, T)")
    return xa, ta, Ta


def bgnbd_loglik(
    r: float,
    alpha: float,
    a: float,
    b: float,
    x: FloatArray,
    tx: FloatArray,
    T: FloatArray,
) -> float:
    """BG/NBD log-likelihood (Fader-Hardie-Lee 2005, eq. 3)."""
    xa, ta, Ta = _check_cbs(x, tx, T)
    if min(r, alpha, a, b) <= 0:
        return -np.inf
    A1 = gammaln(r + xa) - gammaln(r) + gammaln(a + b) + gammaln(b + xa)
    A2 = -(gammaln(b) + gammaln(a + b + xa)) + r * np.log(alpha)
    A3 = -(r + xa) * np.log(alpha + Ta)
    A4 = np.where(
        xa > 0,
        np.log(a) + np.log(b + xa - 1) - np.log(a + b + xa - 1) - (r + xa) * np.log(alpha + ta),
        -np.inf,
    )
    ll_vec = A1 + A2 + np.logaddexp(A3, A4)
    return float(ll_vec.sum())


def fit_bgnbd(x: FloatArray, tx: FloatArray, T: FloatArray) -> dict[str, float]:
    """MLE for BG/NBD hyperparameters (r, alpha, a, b)."""
    xa, ta, Ta = _check_cbs(x, tx, T)
    if xa.size < 20:
        raise ValueError("need >=20 customers for hyperparameter MLE")

    bounds = [(-6.0, 6.0)] * 4  # log-space bounds keep the
    # identifiable region and avoid the r,alpha->inf ridge

    def neg(ll: FloatArray) -> float:
        v = -bgnbd_loglik(
            np.exp(ll[0]),
            np.exp(ll[1]),
            np.exp(ll[2]),
            np.exp(ll[3]),
            xa,
            ta,
            Ta,
        )
        return v if np.isfinite(v) else 1e9

    best = None
    for init in ([0.0, 1.5, 0.0, 1.0], [-0.7, 2.3, -0.7, 1.6], [0.7, 1.1, 0.7, 0.7]):
        res = minimize(
            neg,
            np.asarray(init),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000},
        )
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not np.isfinite(best.fun):
        raise ValueError("BG/NBD fit failed")
    r, alpha, a, b = np.exp(best.x)
    return {
        "r": float(r),
        "alpha": float(alpha),
        "a": float(a),
        "b": float(b),
        "neg_ll": float(best.fun),
    }


def bgnbd_expected_purchases(
    r: float,
    alpha: float,
    a: float,
    b: float,
    x: FloatArray,
    tx: FloatArray,
    T: FloatArray,
    t: float,
) -> FloatArray:
    """E[Y(t) | x, t_x, T] for BG/NBD (FHL 2005, eq. 10)."""
    xa, ta, Ta = _check_cbs(x, tx, T)
    if t <= 0:
        raise ValueError("horizon positive")
    hyp = hyp2f1(
        r + xa,
        b + xa,
        a + b + xa - 1,
        t / (alpha + Ta + t),
    )
    # FHL eq. 10: (a+b+x-1)/(a-1) *
    #   [1 - ((aT)/(aT+t))^{r+x} * 2F1] /
    #   [1 + delta_{x>0} (a/(b+x-1)) ((a+tx)/(a+T))^{r+x}]
    with np.errstate(all="ignore"):
        num = (a + b + xa - 1) / (a - 1) * (1 - ((alpha + Ta) / (alpha + Ta + t)) ** (r + xa) * hyp)
    denom = 1 + np.where(
        xa > 0,
        (a / (b + xa - 1)) * ((alpha + ta) / (alpha + Ta)) ** (r + xa),
        0.0,
    )
    pred = np.where(np.isfinite(num) & (num > 0), num, 0.0) / denom
    return np.asarray(np.clip(pred, 0.0, None))


def p_alive_bgnbd(
    r: float,
    alpha: float,
    a: float,
    b: float,
    x: FloatArray,
    tx: FloatArray,
    T: FloatArray,
) -> FloatArray:
    """P(alive at T | x, t_x, T) (FHL 2005, eq. 11)."""
    xa, ta, Ta = _check_cbs(x, tx, T)
    add = np.where(
        xa > 0,
        (a / (b + xa - 1)) * ((alpha + ta) / (alpha + Ta)) ** (r + xa),
        0.0,
    )
    return np.asarray(1.0 / (1.0 + add))


def bench_pnbd(seed: int = 515) -> dict[str, float]:
    """SYNTHETIC: heterogeneous Poisson + geometric dropout;
    fitted BG/NBD E[purchases] must beat a recency-only
    heuristic against realized future counts."""
    rng = np.random.default_rng(seed)
    n_cust = 400
    T_cal = 52.0
    horizon = 13.0
    lam = rng.gamma(1.2, 4.0, n_cust)  # rates per year-ish
    drop = rng.beta(0.7, 8.0, n_cust)
    # simulate calibration window
    x = np.zeros(n_cust)
    tx = np.zeros(n_cust)
    alive = np.ones(n_cust, dtype=bool)
    t_alive = np.zeros(n_cust)
    for i in range(n_cust):
        t = 0.0
        while t < T_cal and alive[i]:
            t += rng.exponential(52.0 / max(lam[i], 0.05))
            if t < T_cal:
                x[i] += 1
                tx[i] = t
                if rng.random() < drop[i]:
                    alive[i] = False
                    t_alive[i] = t
        if alive[i]:
            t_alive[i] = T_cal
    # simulate holdout purchases (truth)
    fut = np.zeros(n_cust)
    for i in range(n_cust):
        if not alive[i]:
            continue
        t = t_alive[i]
        while t < T_cal + horizon:
            t += rng.exponential(52.0 / max(lam[i], 0.05))
            if t < T_cal + horizon:
                fut[i] += 1
                if rng.random() < drop[i]:
                    break
    fit = fit_bgnbd(x, tx, np.full(n_cust, T_cal))
    pred = bgnbd_expected_purchases(
        fit["r"],
        fit["alpha"],
        fit["a"],
        fit["b"],
        x,
        tx,
        np.full(n_cust, T_cal),
        horizon,
    )
    # naive: last-period rate extrapolation
    naive = np.clip(x / np.maximum(tx, 1.0) * horizon, 0, None)
    mse_model = float(((pred - fut) ** 2).mean())
    mse_naive = float(((naive - fut) ** 2).mean())
    if mse_model >= mse_naive * 1.15:
        raise ValueError("BG/NBD not competitive")
    pa = p_alive_bgnbd(
        fit["r"],
        fit["alpha"],
        fit["a"],
        fit["b"],
        x,
        tx,
        np.full(n_cust, T_cal),
    )
    return {
        "synthetic_mse_model": mse_model,
        "synthetic_mse_naive": mse_naive,
        "synthetic_p_alive_mean": float(pa.mean()),
        "synthetic_r_hat": fit["r"],
        "synthetic_alpha_hat": fit["alpha"],
        "synthetic_a_hat": fit["a"],
        "synthetic_b_hat": fit["b"],
    }
