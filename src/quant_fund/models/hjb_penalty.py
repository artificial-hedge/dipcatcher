"""Penalty-method HJB solver for the American put (semi-smooth iteration) (SYNTHETIC).

Backward Euler with penalty ρ·max(payoff − V, 0): each step solves
(I/dt − A + ρ·diag(1_{V<f})) V = V_prev/dt + ρ·1_{V<f}·f by a few
semi-smooth Newton sweeps of the tridiagonal system.
"""

import numpy as np

from quant_fund.models._amopt_synth import S0, SIG, K, Q, R, T, crr_price, european_put


def _thomas(lo: np.ndarray, di: np.ndarray, up: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    n = len(di)
    cp, dp = np.zeros(n), np.zeros(n)
    cp[0], dp[0] = up[0] / di[0], rhs[0] / di[0]
    for i in range(1, n):
        m = di[i] - lo[i] * cp[i - 1]
        cp[i] = up[i] / m
        dp[i] = (rhs[i] - lo[i] * dp[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _penalty_put(ns: int = 200, nt: int = 120, rho: float = 1e7) -> float:
    smax = 4.0 * K
    dt = T / nt
    s = np.linspace(0.0, smax, ns + 1)
    i = np.arange(ns + 1)
    lo = 0.5 * SIG**2 * i**2 - 0.5 * (R - Q) * i
    di = -(SIG**2 * i**2 + R)
    up = 0.5 * SIG**2 * i**2 + 0.5 * (R - Q) * i
    payoff = np.maximum(K - s, 0.0)
    v = payoff.copy()
    for _ in range(nt):
        rhs0 = v / dt
        y = v.copy()
        for _it in range(8):
            act = y < payoff
            mlo, mdi, mup = -dt * lo, 1.0 - dt * di + rho * dt * act, -dt * up
            mrhs = rhs0 * dt + rho * dt * act * payoff
            mlo[0], mdi[0], mup[0], mrhs[0] = 0.0, 1.0, 0.0, K
            mlo[-1], mdi[-1], mup[-1], mrhs[-1] = 0.0, 1.0, 0.0, 0.0
            loi, upi = mlo.copy(), mup.copy()
            loi[1:] = mlo[1:]
            upi[:-1] = mup[:-1]
            y = _thomas(loi, mdi, upi, mrhs)
        v = y
    return float(np.interp(S0, s, v))


def bench_hjb_penalty(seed: int = 4107) -> dict[str, float]:
    del seed
    ref, _ = crr_price(2000)
    price = _penalty_put()
    eur = european_put()
    return {
        "synthetic_hjb_price": price,
        "synthetic_hjb_err": abs(price - ref),
        "synthetic_hjb_ref": ref,
        "synthetic_hjb_eur_floor_err": abs(eur - ref),
    }
