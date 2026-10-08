"""Berry-Levinsohn-Pakes (1995) random-coefficients logit demand (SYNTHETIC).

BLP demand separates the mean utility δ_jt (inverted from
observed shares by contraction mapping) from the random-
coefficients variance σ. Identification: the moment
E[ξ_jt · z_jt] = 0 over product/marker instruments — here a
two-step GMM on (δ = Xβ − αp + ξ) with an outer σ search.

Honesty: synthetic markets simulate shares from known
(β, α, σ); the bench inverts and re-estimates them — proper
diagnostics, never market evidence.

References:
- Berry, S., Levinsohn, J., Pakes, A. (1995). Automobile
  prices in market equilibrium. *Econometrica* 63 — the
  model, share inversion and BLP instruments.
- Nevo, A. (2000). A practitioner's guide to estimation of
  random-coefficients logit models of demand. *Journal of
  Economics & Management Strategy* 9 — the contraction and
  GMM steps used here.
- Berry, S. (1994). Estimating discrete-choice models of
  product differentiation. *RAND Journal of Economics* 25 —
  inversion existence.
- Dubé, J.-P., Fox, J. T., Su, C.-L. (2012). Improving the
  numerical performance of static and dynamic aggregate
  discrete choice random coefficients demand estimation.
  *Econometrica* 80 — tolerance/consistency issues (the
  tight contraction tolerance below).

Composition: numpy + scipy.optimize only — contraction
mapping + inner linear GMM; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _share_pred(
    delta: FloatArray,
    x_rc: FloatArray,
    nu: FloatArray,
    sigma: float,
    mkt: FloatArray,
) -> FloatArray:
    """Predicted shares: for each market, s_j = mean_i
    exp(δ_j + σ·ν_i·x_j)/Σ_k exp(...), plus outside option 0."""
    mkts = np.unique(mkt)
    pred = np.zeros(delta.size)
    for m in mkts:
        sel = mkt == m
        d = delta[sel]
        xr = x_rc[sel]
        # (n_i, n_j) individual choice probabilities
        u = d[None, :] + sigma * nu[:, None] * xr[None, :]
        eu = np.exp(u - u.max(axis=1, keepdims=True))
        denom = np.exp(-u.max(axis=1)) + eu.sum(axis=1)
        pred[sel] = (eu / denom[:, None]).mean(axis=0)
    return np.asarray(pred, dtype=np.float64)


def blp_invert(
    shares: FloatArray,
    x_rc: FloatArray,
    nu: FloatArray,
    sigma: float,
    mkt: FloatArray,
    tol: float = 1e-10,
    max_iter: int = 20000,
) -> FloatArray:
    """Contraction-map δ: δ_{t+1} = δ_t + ln s_obs − ln s_pred(δ)."""
    s = np.asarray(shares, dtype=np.float64)
    if s.ndim != 1 or not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("positive finite shares required")
    if np.any(np.isclose(s, 1.0)):
        raise ValueError("shares must be < 1 (outside option)")
    if nu.ndim != 1 or nu.size < 10:
        raise ValueError("nu: 1-D draws, >=10 required")
    if mkt.shape != s.shape or x_rc.shape != s.shape:
        raise ValueError("shape mismatch")
    # start δ at ln s_j − ln s0 (s0 = market outside share)
    delta = np.log(s)
    for m in np.unique(mkt):
        sel = mkt == m
        s0 = max(1e-12, 1.0 - float(np.sum(s[sel])))
        delta[sel] -= np.log(s0)
    for _ in range(max_iter):
        pred = _share_pred(delta, x_rc, nu, sigma, mkt)
        nxt = delta + np.log(s) - np.log(pred)
        if np.max(np.abs(nxt - delta)) < tol:
            return np.asarray(nxt, dtype=np.float64)
        delta = nxt
        # closed-form Newton step on the slow common-shift
        # eigenvector: per market, Σpred(δ+c) ≈ Σpred +
        # c·s_in·s_out for uniform logit shifts — solves the
        # inside-share gap in one step.
        pred = _share_pred(delta, x_rc, nu, sigma, mkt)
        for m in np.unique(mkt):
            sel = mkt == m
            s_in = float(np.sum(pred[sel]))
            gap = float(np.sum(s[sel])) - s_in
            denom = s_in * max(1e-6, 1.0 - s_in)
            if abs(denom) > 1e-12:
                delta[sel] += float(np.clip(gap / denom, -5.0, 5.0))
    raise ValueError("share inversion did not converge")


def _gmm_linear(
    delta: FloatArray,
    x: FloatArray,
    price: FloatArray,
    z: FloatArray,
) -> tuple[FloatArray, float, FloatArray]:
    """Two-stage least squares form of the BLP linear moment:
    δ = Xβ − αp + ξ, instruments [X, Z]. Returns (params,
    obj, xi)."""
    n = delta.size
    reg = np.column_stack([x, price])
    w = np.column_stack([x, z])  # instruments: exogenous x + cost shifters
    pz = w @ np.linalg.pinv(w.T @ w) @ w.T
    beta = np.linalg.solve(reg.T @ pz @ reg, reg.T @ pz @ delta)
    xi = delta - reg @ beta
    obj = float((xi @ w @ w.T @ xi) / n)
    return np.asarray(beta, dtype=np.float64), obj, np.asarray(xi, dtype=np.float64)


def blp_estimate(
    shares: FloatArray,
    x: FloatArray,
    price: FloatArray,
    x_rc: FloatArray,
    z: FloatArray,
    mkt: FloatArray,
    n_draws: int = 60,
    sigma_grid: FloatArray | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Estimate (β, α, σ) — σ by grid search over the GMM
    objective (one random coefficient on x_rc)."""
    s = np.asarray(shares, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    p = np.asarray(price, dtype=np.float64)
    xr = np.asarray(x_rc, dtype=np.float64)
    zz = np.asarray(z, dtype=np.float64)
    mm = np.asarray(mkt, dtype=np.float64)
    n = s.size
    if xx.ndim != 2 or xx.shape[0] != n or p.shape != (n,) or xr.shape != (n,):
        raise ValueError("shape mismatch")
    if zz.ndim == 1:
        zz = zz[:, None]
    if zz.shape[0] != n or mm.shape != (n,):
        raise ValueError("shape mismatch")
    if not (
        np.all(np.isfinite(xx))
        and np.all(np.isfinite(p))
        and np.all(np.isfinite(zz))
        and np.all(np.isfinite(s))
    ):
        raise ValueError("finite inputs required")
    rng = np.random.default_rng(seed)
    nu = rng.standard_normal(n_draws)
    # BLP instruments for the random coefficient: rival
    # characteristic sums Σ_{k≠j} x_rc,k within each market.
    z_rival = np.zeros(n)
    for m in np.unique(mm):
        sel = mm == m
        z_rival[sel] = float(np.sum(xr[sel])) - xr[sel]
    zz = np.column_stack([zz, z_rival])
    grid = (
        np.asarray(sigma_grid, dtype=np.float64)
        if sigma_grid is not None
        else np.linspace(0.0, 2.5, 26)
    )
    best: dict[str, object] | None = None
    best_obj = float("inf")
    for sg in grid:
        try:
            delta = blp_invert(s, xr, nu, float(sg), mm)
        except ValueError:
            continue
        beta, obj, xi = _gmm_linear(delta, xx, p, zz)
        if best is None or obj < best_obj:
            best_obj = obj
            best = {
                "beta": beta[:-1],
                "alpha": -float(beta[-1]),
                "sigma": float(sg),
                "obj": obj,
                "xi": xi,
            }
    if best is None:
        raise ValueError("no convergent sigma")
    return best


def synth_blp_market(
    n_markets: int = 20,
    n_products: int = 15,
    beta1: float = 1.5,
    alpha: float = 1.0,
    sigma: float = 1.0,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Simulate shares from u_ijt = β x_j − α p_j + σ ν_i x_j
    + ξ_j + ε_ij (outside option normalized 0)."""
    rng = np.random.default_rng(seed)
    n = n_markets * n_products
    mkt = np.repeat(np.arange(n_markets), n_products).astype(np.float64)
    x = rng.uniform(0.5, 2.0, n)
    cost = rng.uniform(0.2, 1.0, n)
    z = rng.normal(0, 1, n)  # extra cost-shifter instrument
    p = 1.0 + 0.5 * cost + 0.3 * z + rng.normal(0, 0.2, n)
    xi = rng.normal(0, 0.5, n)
    n_draws = 200
    nu = rng.standard_normal(n_draws)
    shares = np.zeros(n)
    for m in np.unique(mkt):
        sel = mkt == m
        d = beta1 * x[sel] - alpha * p[sel] + xi[sel]
        u = d[None, :] + sigma * nu[:, None] * x[sel][None, :]
        eu = np.exp(u)
        denom = 1.0 + eu.sum(axis=1)
        shares[sel] = (eu / denom[:, None]).mean(axis=0)
    return {
        "shares": shares,
        "x": np.column_stack([np.ones(n), x]),
        "price": p,
        "x_rc": x,
        "z": np.column_stack([cost, z]),
        "mkt": mkt,
    }


def bench_blp_demand(seed: int = 20261231 + 263) -> dict[str, float]:
    """BLP self-check: recovers α≈1 and σ≈1 from simulated
    shares where plain logit (σ=0) is price-biased.
    All ``synthetic_*``."""
    d = synth_blp_market(alpha=1.0, sigma=1.0, seed=seed)
    out = blp_estimate(
        np.asarray(d["shares"]),
        np.asarray(d["x"]),
        np.asarray(d["price"]),
        np.asarray(d["x_rc"]),
        np.asarray(d["z"]),
        np.asarray(d["mkt"]),
        n_draws=60,
        seed=seed + 1,
    )
    # plain-logit α for comparison (σ=0, OLS on inversion)
    s = np.asarray(d["shares"])
    d0 = np.log(s) - np.log(1 - s)
    reg = np.column_stack([np.asarray(d["x"]), np.asarray(d["price"])])
    b0 = np.linalg.lstsq(reg, d0, rcond=None)[0]
    alpha_naive = -float(b0[-1])
    out2 = blp_estimate(
        np.asarray(d["shares"]),
        np.asarray(d["x"]),
        np.asarray(d["price"]),
        np.asarray(d["x_rc"]),
        np.asarray(d["z"]),
        np.asarray(d["mkt"]),
        n_draws=60,
        seed=seed + 1,
    )
    alpha_hat = out["alpha"]
    sigma_hat = out["sigma"]
    if not (isinstance(alpha_hat, float) and isinstance(sigma_hat, float)):
        raise ValueError("isinstance(alpha_hat, float) and isinstance(sigma_hat, float)")
    return {
        "synthetic_alpha": alpha_hat,
        "synthetic_sigma": sigma_hat,
        "synthetic_alpha_true": 1.0,
        "synthetic_sigma_true": 1.0,
        "synthetic_alpha_naive_logit": alpha_naive,
        "synthetic_detects": float(abs(alpha_hat - 1.0) < 0.15 and abs(sigma_hat - 1.0) < 0.3),
        "synthetic_determinism": float(out2["sigma"] == sigma_hat),
    }
