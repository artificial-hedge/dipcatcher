"""Multiplicative-weights (Hedge) regret vs the halving bound (SYNTHETIC).

Hedge with eta = sqrt(ln K / T) achieves E-regret <= 2 sqrt(T ln K).
Bench: simulated Hedge loss vs best fixed expert, regret curve vs the
2 sqrt(T ln K) envelope at several horizons.
"""

import numpy as np

from quant_fund.models._rlt_synth import EXP_P, T_EXP


def _hedge(seed: int, t_max: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = len(EXP_P)
    losses = (rng.random((t_max, k)) > EXP_P).astype(np.float64)  # Bernoulli losses
    eta = np.sqrt(np.log(k) / t_max)
    w = np.ones(k)
    hedge_loss = np.zeros(t_max)
    for t in range(t_max):
        p = w / w.sum()
        hedge_loss[t] = p @ losses[t]
        w *= np.exp(-eta * losses[t])
    return hedge_loss, losses


def bench_mw_hedge(seed: int = 4603) -> dict[str, float]:
    hl, losses = _hedge(seed, T_EXP)
    cum_h = np.cumsum(hl)
    t = np.arange(1, T_EXP + 1)
    best_e = 1.0 - EXP_P.max()  # E[loss] of best expert
    pseudo = cum_h - t * best_e
    cum_best = np.cumsum(losses, axis=0).min(axis=1)
    realized = cum_h - cum_best
    bnd = 2.0 * np.sqrt(t * np.log(len(EXP_P)))
    return {
        "synthetic_mw_pseudo": float(pseudo[-1]),
        "synthetic_mw_bound": float(bnd[-1]),
        "synthetic_mw_margin": float(bnd[-1] - pseudo[-1]),
        "synthetic_mw_realized": float(realized[-1]),
        "synthetic_mw_rate": float(pseudo[-1] / np.sqrt(T_EXP)),
    }
