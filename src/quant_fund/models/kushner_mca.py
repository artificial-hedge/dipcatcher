"""Kushner Markov-chain approximation for the American-put stochastic control (SYNTHETIC).

Log-price grid with trinomial transitions matched to the local first two
moments of the GBM drift/diffusion (Kushner's local-consistency
construction); backward dynamic programming applies the max(payoff,
continuation) control at every node.
"""

import numpy as np

from quant_fund.models._amopt_synth import S0, SIG, K, Q, R, T, crr_price, european_put


def _mca_put(nx: int = 160, nt: int = 120) -> float:
    x0 = np.log(S0)
    h = SIG * np.sqrt(3.0 * T / nt)
    dt = T / nt
    xs = x0 + h * np.arange(-nx // 2, nx // 2 + 1)
    mu = R - Q - 0.5 * SIG**2
    pu = 0.5 * (SIG**2 * dt / h**2 + mu * dt / h)
    pd = 0.5 * (SIG**2 * dt / h**2 - mu * dt / h)
    pm = max(0.0, 1.0 - pu - pd)
    disc = np.exp(-R * dt)
    payoff = np.maximum(K - np.exp(xs), 0.0)
    v = payoff.copy()
    for _ in range(nt):
        cont = disc * (pu * np.roll(v, -1) + pm * v + pd * np.roll(v, 1))
        cont[0], cont[-1] = K, 0.0
        v = np.maximum(payoff, cont)
    return float(v[nx // 2])


def bench_kushner_mca(seed: int = 4105) -> dict[str, float]:
    del seed
    ref, _ = crr_price(2000)
    price = _mca_put()
    eur = european_put()
    return {
        "synthetic_mca_price": price,
        "synthetic_mca_err": abs(price - ref),
        "synthetic_mca_ref": ref,
        "synthetic_mca_eur_floor_err": abs(eur - ref),
    }
