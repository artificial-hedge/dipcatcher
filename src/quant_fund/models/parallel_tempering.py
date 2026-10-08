"""Parallel tempering (replica exchange) MCMC on a bimodal 1-D target (SYNTHETIC).

K replicas at temperatures beta_k = 2^{-k}; within-replica random-walk
MH plus adjacent-replica swap moves with the PT acceptance ratio.
Bench compares the coldest chain's mode-mass estimate against the
analytic value and a single-chain baseline that gets stuck.
"""

import numpy as np


def _logp(x: float) -> float:
    # bimodal: small mode at -4 (weight 0.3), big at +2 (weight 0.7)
    return float(
        np.logaddexp(
            np.log(0.3) - 0.5 * ((x + 4.0) / 0.6) ** 2,
            np.log(0.7) - 0.5 * ((x - 2.0) / 0.8) ** 2,
        )
    )


def _pt(seed: int, n: int = 4000, k: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    betas = 2.0 ** -np.arange(k)
    xs = rng.normal(0.0, 3.0, k)
    out = np.zeros(n)
    for t in range(n):
        for i in range(k):
            prop = xs[i] + rng.normal(0, 1.0)
            if rng.random() < np.exp(betas[i] * (_logp(prop) - _logp(xs[i]))):
                xs[i] = prop
        # adjacent swaps
        for i in range(k - 1):
            d = (betas[i] - betas[i + 1]) * (_logp(xs[i + 1]) - _logp(xs[i]))
            if rng.random() < np.exp(d):
                xs[i], xs[i + 1] = xs[i + 1], xs[i]
        out[t] = xs[0]
    return out


def _single(seed: int, n: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 3.0)
    out = np.zeros(n)
    for t in range(n):
        prop = x + rng.normal(0, 1.0)
        if rng.random() < np.exp(_logp(prop) - _logp(x)):
            x = prop
        out[t] = x
    return out


def bench_parallel_tempering(seed: int = 5601) -> dict[str, float]:
    pt = _pt(seed)
    sc = _single(seed)
    mass_true = 0.3  # left-mode probability mass
    pt_mass = float(np.mean(pt < -1.0))
    sc_mass = float(np.mean(sc < -1.0))
    return {
        "synthetic_pt_mass": pt_mass,
        "synthetic_pt_sc_mass": sc_mass,
        "synthetic_pt_true": mass_true,
        "synthetic_pt_err": abs(pt_mass - mass_true),
        "synthetic_pt_sc_err": abs(sc_mass - mass_true),
    }
