"""Track-and-Stop (Garivier & Kaufmann 2016) — asymptotically (SYNTHETIC)
optimal fixed-confidence BAI: forced exploration toward the
Chernoff-oracle weights ω_i ∝ 1/Δ_i² (Gaussian relaxation), GLR
stopping at β(t) = log((log t + 1)/δ)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.lil_ucb import GaussianBandit

FloatArray = NDArray[np.float64]


def oracle_weights(means: FloatArray, sigma: float = 1.0) -> FloatArray:
    """Gaussian oracle weights: ω_i ∝ 1/(μ*−μ_i)² for i≠*, and
    ω_* = 1 − Σ ω_i pinned so that T_i·Δ_i² balanced (inverse-gap
    square allocation, the standard TnS approximation)."""
    k = len(means)
    star = int(np.argmax(means))
    w = np.zeros(k)
    gaps = np.maximum(means[star] - means, 1e-3)
    inv = 1.0 / gaps**2
    inv[star] = 0.0
    s_inv = float(inv.sum())
    if k > 1 and s_inv > 0:
        # Gaussian TnS oracle: equalize per-constraint rates
        # w_*·Δ_min² = w_i·Δ_i² → w_* = 1/(1 + S·Δ_min²)
        # with S = Σ_{i≠*} Δ_i^{-2}; challengers split ∝ 1/Δ_i².
        d_min2 = float(gaps[np.arange(k) != star].min() ** 2)
        w_star = 1.0 / (1.0 + s_inv * d_min2)
        w[star] = w_star
        w += inv * ((1.0 - w_star) / s_inv)
    else:
        w[star] = 1.0
    out: FloatArray = np.asarray(w / w.sum())
    return out


def track_stop(
    mu: FloatArray,
    rng: np.random.Generator,
    delta: float = 0.1,
    sigma: float = 1.0,
    max_pulls: int = 60000,
) -> tuple[int, int]:
    """TnS: forced tracking of oracle weights (recomputed each
    round) + C-tracking (pull the most under-sampled arm relative to
    current weights)."""
    env = GaussianBandit(mu, sigma)
    k = env.k
    for i in range(k):
        env.pull(i, rng)
    t = k
    while t < max_pulls:
        means = env.means()
        t_ns = env.counts
        star = int(np.argmax(means))
        z = np.inf
        for i in range(k):
            if i == star:
                continue
            num = t_ns[i] * t_ns[star] * (means[star] - means[i]) ** 2
            den = 2.0 * sigma**2 * (t_ns[i] + t_ns[star])
            z = min(z, num / den)
        if z > np.log((np.log(t) + 1.0) / delta):
            return star, t
        w = oracle_weights(means, sigma)
        deficits = w * t - t_ns
        pull = int(np.argmax(deficits))
        env.pull(pull, rng)
        t += 1
    return int(np.argmax(env.means())), t


def bench_track_stop(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: TnS identifies the best arm; pulls concentrate on
    the competitive arms (oracle allocation)."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    mu = np.array([0.0, 0.3, 0.15, 0.25])
    best, pulls = track_stop(mu, rng, delta=0.1)
    out["synthetic_tns_correct"] = float(best == 1)
    out["synthetic_tns_pulls"] = float(pulls)
    out["synthetic_tns_budget_ok"] = float(pulls < 60000)
    w = oracle_weights(np.array([0.0, 0.3, 0.15]))
    out["synthetic_tns_weights_sum"] = float(w.sum())
    out["synthetic_tns_weights_valid"] = float(np.isfinite(w).all() and (w >= 0).all())
    return out


if __name__ == "__main__":
    print(bench_track_stop())
