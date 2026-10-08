"""Synthetic American-option fixture shared by the stochastic-control canon (SYNTHETIC).

A Black-Scholes American put contract plus a fine-grid reference price
computed by a converged binomial tree. Every module in the wave prices the
same contract so errors are measured against one honest reference, with the
European put as the always-lower honest-negative baseline.
"""

import numpy as np

S0, K, T, R, Q, SIG = 100.0, 105.0, 0.5, 0.05, 0.02, 0.25


def amopt_params() -> dict[str, float]:
    return {"S0": S0, "K": K, "T": T, "r": R, "q": Q, "sigma": SIG}


def gbm_path(seed: int, steps: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = T / steps
    z = rng.normal(size=steps)
    logS = np.log(S0) + np.cumsum((R - Q - 0.5 * SIG**2) * dt + SIG * np.sqrt(dt) * z)
    return np.asarray(np.exp(logS), dtype=np.float64)


def european_put() -> float:
    from math import erf, exp, log, sqrt

    d1 = (log(S0 / K) + (R - Q + 0.5 * SIG**2) * T) / (SIG * sqrt(T))
    d2 = d1 - SIG * sqrt(T)

    def nd(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    return float(K * exp(-R * T) * nd(-d2) - S0 * exp(-Q * T) * nd(-d1))


def crr_price(n: int = 2000) -> tuple[float, np.ndarray]:
    """Reference CRR binomial American-put price + exercise boundary per step."""
    dt = T / n
    u = np.exp(SIG * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((R - Q) * dt) - d) / (u - d)
    disc = np.exp(-R * dt)
    j = np.arange(n + 1)
    S = S0 * u**j * d ** (n - j)
    V = np.maximum(K - S, 0.0)
    boundary = np.full(n + 1, K)
    for t in range(n - 1, -1, -1):
        j = np.arange(t + 1)
        S = S0 * u**j * d ** (t - j)
        cont = disc * (p * V[1 : t + 2] + (1 - p) * V[: t + 1])
        ex = np.maximum(K - S, 0.0)
        V = np.maximum(ex, cont)
        exers = np.where(ex > cont)[0]
        boundary[t] = float(S[exers[-1]]) if len(exers) else K
    return float(V[0]), boundary
