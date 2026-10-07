"""Deep hedging: learned delta-shrinkage policy vs raw Black-Scholes (SYNTHETIC)
delta under proportional transaction costs on synthetic GBM paths.

Policy = clip(a * bs_delta + b * tau_decay, 0, 1): a 2-parameter
"trained" hedge that damps turnover near expiry. Chosen by CVaR10 of
hedged terminal P&L on the training path ensemble, evaluated on a
fresh ensemble.
"""

import numpy as np
from scipy.stats import norm


def _bs_delta(s: np.ndarray, k: float, tau, sig: float) -> np.ndarray:
    tau = np.maximum(np.asarray(tau, dtype=float), 1e-6)
    d = (np.log(s / k) + 0.5 * sig**2 * tau) / (sig * np.sqrt(tau))
    return np.asarray(norm.cdf(d))


def _hedge(paths: np.ndarray, a: float, b: float, cost: float) -> np.ndarray:
    k, sig, prem = 100.0, 0.25, 5.0
    n, steps = paths.shape[0], paths.shape[1] - 1
    dt = 1.0 / steps
    wealth = np.full(n, prem)
    d_prev = np.zeros(n)
    for t in range(steps):
        s = paths[:, t]
        tau = 1.0 - t * dt
        d = np.clip(a * _bs_delta(s, k, tau, sig) + b * np.exp(-tau * 4.0), 0.0, 1.0)
        wealth -= cost * s * np.abs(d - d_prev)
        wealth += d * (paths[:, t + 1] - s)
        d_prev = d
    wealth -= np.maximum(paths[:, -1] - k, 0)
    return wealth


def _mk_paths(rng: np.random.Generator, n: int, steps: int) -> np.ndarray:
    dt = 1.0 / steps
    z = rng.normal(0, np.sqrt(dt), (n, steps))
    return 100.0 * np.exp(
        np.concatenate([np.zeros((n, 1)), np.cumsum(-0.5 * 0.25**2 * dt + 0.25 * z, 1)], 1)
    )


def bench_deep_hedge(seed: int = 5805) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    train = _mk_paths(rng, 4000, 20)
    test = _mk_paths(rng, 4000, 20)
    cost = 0.002
    best, bw = -np.inf, (1.0, 0.0)
    for a in np.linspace(0.7, 1.0, 13):
        for b in np.linspace(-0.2, 0.2, 9):
            cvar = np.quantile(_hedge(train, a, b, cost), 0.1)
            if cvar > best:
                best, bw = cvar, (float(a), float(b))
    pnl_pol = _hedge(test, *bw, cost)
    pnl_bs = _hedge(test, 1.0, 0.0, cost)
    pnl_bs0 = _hedge(test, 1.0, 0.0, 0.0)  # costless reference
    return {
        "synthetic_dh_cvar": float(np.quantile(pnl_pol, 0.1)),
        "synthetic_dh_bs_cvar": float(np.quantile(pnl_bs, 0.1)),
        "synthetic_dh_bs0_cvar": float(np.quantile(pnl_bs0, 0.1)),
        "synthetic_dh_a": bw[0],
        "synthetic_dh_b": bw[1],
        "synthetic_dh_gain": float(np.quantile(pnl_pol, 0.1) > np.quantile(pnl_bs, 0.1)),
    }
