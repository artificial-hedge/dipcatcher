"""Nested logit — McFadden two-level choice with correlated utilities.

Alternatives sit inside nests; errors share a nest component, so IIA
holds within a nest but not across nests. The inclusive value
λ governs within-nest correlation: λ→1 collapses to multinomial
logit (IIA everywhere); λ<1 admits substitution concentrated
inside nests. Estimated by full-information ML on the nested logit
likelihood:

  P(i|nest k) ∝ exp(x_iβ/λ_k)
  P(nest k)  ∝ exp(λ_k · IV_k),  IV_k = log Σ_{j∈k} exp(x_jβ/λ_k)

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure parameter recovery on generated
GEV-correlated utilities — never market evidence.

References:
- McFadden, D. (1978). Modelling the choice of residential location.
  In *Spatial Interaction Theory and Planning Models* — the nested
  logit / GEV model.
- McFadden, D. (1981). Econometric models of probabilistic choice.
  In *Structural Analysis of Discrete Data* — FIML estimation and
  the inclusive-value coefficient.
- Train, K. E. (2009). *Discrete Choice Methods with Simulation*,
  2nd ed., ch. 4 — the λ∈(0,1] consistency bound.
- Cardell, N. S., Dunbar, F. C. (1980). Measuring the societal
  impacts of automobile downsizing. *Transportation Research A* 14.

Composition: pure numpy + scipy — joint FIML via L-BFGS-B on
(β, log-λ), sequential MNL start; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import logsumexp

FloatArray = NDArray[np.float64]


def nested_logit_fit(
    choice: FloatArray,
    x: FloatArray,
    nest: FloatArray,
) -> dict[str, float]:
    """Two-level nested logit FIML.

    ``choice[i]`` is the observed alternative for chooser i (0..J-1);
    ``x`` is (n_choosers × n_alt × n_attr) so
    ``x[i, j, :]`` are attributes of alternative j for chooser i;
    ``nest[j]`` is alternative j's nest (0..K-1)."""
    xx3 = np.asarray(x, dtype=np.float64)
    if xx3.ndim != 3:
        raise ValueError("x must be (choosers × alternatives × attributes)")
    n_i, n_alt, k = xx3.shape
    c = np.asarray(choice, dtype=np.float64).ravel()
    ns = np.asarray(nest, dtype=np.float64).ravel().astype(int)
    if c.size != n_i or n_i < 60:
        raise ValueError("choice length must match choosers, n>=60")
    if ns.size != n_alt:
        raise ValueError("nest length must equal n alternatives")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(xx3)):
        raise ValueError("finite inputs required")
    if np.any(c < 0) or np.any(c > n_alt - 1):
        raise ValueError("choice ids out of range")
    if np.unique(ns).size < 2:
        raise ValueError("need >=2 nests")
    if k < 1 or k > 5:
        raise ValueError("need 1..5 attributes")
    if np.any(ns < 0) or np.any(ns > np.unique(ns).max()):
        raise ValueError("nest ids out of range")
    nest_ids = np.unique(ns)
    if nest_ids.size != int(nest_ids.max()) + 1:
        raise ValueError("nest ids must be contiguous 0..K-1")

    ci = c.astype(int)

    def nll(theta: FloatArray) -> float:
        beta = theta[:k]
        lam = float(np.exp(theta[k]))  # λ = exp(log-λ), clipped later
        lam = min(max(lam, 0.05), 1.0)
        v = xx3 @ beta  # (i, j)
        # log p_j = log p_j|k + log p_k where
        #   log p_j|k = v_j/λ - IV_k,  log p_k = λ·IV_k - logsumexp_k'(λ·IV_k')
        # with IV_k = log Σ_{j∈k} exp(v_j/λ).
        # where lse_k over chosen alt's nest, lse_K over nests
        iv_all = np.stack(
            [logsumexp(v[:, ns == kk] / lam, axis=1) for kk in nest_ids], axis=1
        )  # (i, K)
        v_ci = v[np.arange(n_i), ci]
        # chosen alt's nest per obs
        nest_of_ci = ns[ci]
        iv_ci = iv_all[np.arange(n_i), nest_of_ci]
        ll_within = v_ci / lam - iv_ci
        lse_nest = logsumexp(lam * iv_all, axis=1)
        ll_nest = lam * iv_ci - lse_nest
        return -float(np.mean(ll_within + ll_nest))

    # start: plain MNL (λ=1 → log-λ=0)
    theta0 = np.concatenate([np.zeros(k), [0.0]])
    res = minimize(
        nll,
        theta0,
        method="L-BFGS-B",
        bounds=[(None, None)] * k + [(-3.0, 0.0)],
        options={"maxiter": 400},
    )
    th = res.x
    beta = th[:k]
    lam = float(np.clip(np.exp(th[k]), 0.05, 1.0))

    # reference MNL (λ=1): log-likelihood
    def mnl_ll(beta_v: FloatArray) -> float:
        vv = xx3 @ beta_v
        return float(np.mean(vv[np.arange(n_i), ci] - logsumexp(vv, axis=1)))

    res2 = minimize(lambda b: -mnl_ll(b), np.zeros(k), method="L-BFGS-B", options={"maxiter": 300})
    ll_mnl = mnl_ll(res2.x)
    ll_nl = -nll(th)

    return {
        "n": float(n_i),
        "n_alt": float(n_alt),
        "lambda": lam,
        "ll_nested": ll_nl,
        "ll_mnl": ll_mnl,
        "lr_vs_mnl": float(2 * (ll_nl - ll_mnl)),
        **{f"beta_{j}": float(v) for j, v in enumerate(beta)},
        "beta_mnl_0": float(res2.x[0]),
    }


def synth_nested_choice(
    n: int = 800,
    n_alt: int = 6,
    nest_of: tuple[int, ...] = (0, 0, 0, 1, 1, 1),
    beta: float = 1.2,
    rho: float = 0.7,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Nested-logit DGP: utility u_ij = β x_ij + ε_ij where ε has
    the GEV nest structure (shared nest shock + idiosyncratic EV)."""
    rng = np.random.default_rng(seed)
    k_nest = int(max(nest_of) + 1)
    x = rng.normal(0.0, 1.0, (n, n_alt, 1))
    eps = np.zeros((n, n_alt))
    nest_shock = rng.gumbel(0.0, 1.0, (n, k_nest)) * math.sqrt(rho)
    for j in range(n_alt):
        eps[:, j] = nest_shock[:, nest_of[j]] + rng.gumbel(0.0, 1.0, n) * math.sqrt(1 - rho)
    u = beta * x[:, :, 0] + eps
    choice = np.argmax(u, axis=1)
    return {
        "choice": choice.astype(np.float64),
        "x": x,
        "nest": np.array(nest_of, dtype=np.float64),
        "beta_true": np.array([beta]),
    }


def bench_nested_logit(
    seed: int = 20261231 + 223,
) -> dict[str, float]:
    """Nested-logit self-check: β recovered under nest-correlated
    utilities; λ<1 detected vs the MNL IIA benchmark.
    All ``synthetic_*``."""
    d = synth_nested_choice(beta=1.2, rho=0.7, seed=seed)
    out = nested_logit_fit(np.asarray(d["choice"]), np.asarray(d["x"]), np.asarray(d["nest"]))
    d0 = synth_nested_choice(beta=0.0, rho=0.7, seed=seed + 1)
    out0 = nested_logit_fit(np.asarray(d0["choice"]), np.asarray(d0["x"]), np.asarray(d0["nest"]))
    out_b = nested_logit_fit(np.asarray(d["choice"]), np.asarray(d["x"]), np.asarray(d["nest"]))

    b = float(out["beta_0"])
    lam = float(out["lambda"])
    return {
        "synthetic_beta": b,
        "synthetic_beta_err": float(abs(b - 1.2)),
        "synthetic_lambda": lam,
        "synthetic_lr_vs_mnl": float(out["lr_vs_mnl"]),
        "synthetic_ll_gain": float(out["ll_nested"] - out["ll_mnl"]),
        "synthetic_null_beta": float(abs(out0["beta_0"])),
        "synthetic_detects": float(
            abs(b - 1.2) < 0.4 and lam < 0.98 and out["ll_nested"] > out["ll_mnl"]
        ),
        "synthetic_determinism": float(
            b == float(out_b["beta_0"]) and lam == float(out_b["lambda"])
        ),
    }
