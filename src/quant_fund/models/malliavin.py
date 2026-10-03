"""Malliavin calculus: D_t W_T = 1 and integration-by-parts (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _bench_malliavin(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    n, m = 400, 200000
    dt = 1.0 / n
    dw = rng.standard_normal((m, n)) * np.sqrt(dt)
    w = np.cumsum(dw, axis=1)
    wt = w[:, -1]
    # D_t W_T = 1 for t <= T: perturbing increment i shifts W_T by the bump
    eps = 1e-4
    dw2 = dw.copy()
    dw2[:, 7] += eps
    wt_pert = np.cumsum(dw2, axis=1)[:, -1]
    checks.append(abs(np.mean(wt_pert - wt) - eps) < 1e-9)
    # IBP: E[F'(W_T)] = E[F(W_T) W_T] / T for W_T ~ N(0,T), T=1
    # F(x) = x^2: E[2 W] = 0; E[W^2 * W]/1 = E[W^3] = 0
    checks.append(abs(np.mean(2 * wt) - np.mean(wt**3)) < 0.05)
    # F(x) = x^3: E[3 W^2] = 3; E[W^4] = 3 -> matches IBP
    checks.append(abs(np.mean(3 * wt**2) - np.mean(wt**4)) < 0.05)
    # F(x) = exp(x): E[exp(W)] = exp(1/2); E[exp(W) W] = exp(1/2)
    lhs = np.mean(np.exp(wt))
    rhs = np.mean(np.exp(wt) * wt)
    checks.append(abs(lhs - np.exp(0.5)) < 0.02)
    checks.append(abs(rhs - np.exp(0.5)) < 0.05)
    # Malliavin-weight estimator of Greeks: E[V'(W)] vs E[V W]/T ~ same
    f = np.maximum(wt - 0.5, 0.0)  # call payoff
    # E[payoff] for N(0,1): phi(0.5)*1 + ... = phi(-0.5) + 0.5? BS with S=1
    # empirical delta weight: E[1_{W>K} * W/T]? use smooth check only
    checks.append(abs(np.mean(f * wt) - np.mean(wt > 0.5)) < 0.02)
    return float(sum(checks) / len(checks))


def bench_malliavin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_malliavin": _bench_malliavin(seed)}
