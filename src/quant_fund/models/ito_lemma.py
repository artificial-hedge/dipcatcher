"""Ito's lemma verification on GBM log-transform (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _bench_ito_lemma(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    n, m = 500, 40000
    dt = 1.0 / n
    mu, sig = 0.08, 0.3
    dw = rng.standard_normal((m, n)) * np.sqrt(dt)
    s = np.ones((m, n + 1))
    for i in range(n):
        s[:, i + 1] = s[:, i] + mu * s[:, i] * dt + sig * s[:, i] * dw[:, i]
    # Ito: log S_T = log S_0 + (mu - sig^2/2) T + sig W_T -> mean drift mu-sig^2/2
    ldrift = np.mean(np.log(s[:, -1]))
    checks.append(abs(ldrift - (mu - 0.5 * sig**2)) < 0.01)
    # variance of log S_T = sig^2 T
    checks.append(abs(np.var(np.log(s[:, -1])) - sig**2) < 0.02)
    # d(S^2) = S^2 (2mu + sig^2) dt + 2 sig S^2 dW -> E[S_T^2] = exp((2mu+sig^2)T)
    checks.append(abs(np.mean(s[:, -1] ** 2) - np.exp(2 * mu + sig**2)) < 0.05)
    # ito correction term is nonzero: log drift != naive mu
    checks.append(abs(ldrift - mu) > 0.03)
    # empirical W_T = (log S_T - (mu - sig^2/2))/sig is ~N(0,T)
    w_emp = (np.log(s[:, -1]) - (mu - 0.5 * sig**2)) / sig
    checks.append(abs(np.mean(w_emp)) < 0.02 and abs(np.var(w_emp) - 1.0) < 0.05)
    return float(sum(checks) / len(checks))


def bench_ito_lemma(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ito_lemma": _bench_ito_lemma(seed)}
