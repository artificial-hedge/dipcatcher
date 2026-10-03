"""Shared synthetic risk-sensitive bandit-MDP for the wave-137
distributional-RL canon.

States are 4 contexts; two actions share near-equal means but differ in
risk shape: action 0 is tight, action 1 has the same mean but a heavy left
tail. Expected-return maximization is indifferent; a CVaR-aware
distributional agent prefers action 0.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_STATES = 4


def synth_reward_env(n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Return (states, contexts) — states in [0,4), one-hot contexts."""
    s = rng.integers(0, _STATES, n)
    x = np.zeros((n, _STATES))
    x[np.arange(n), s] = 1.0
    return s.astype(np.float64), x


def sample_return(states: FloatArray, actions: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Stochastic returns: action 0 ~ N(0.5,0.1); action 1 ~ same mean with
    a heavy left tail (Pareto-like downside)."""
    r = 0.5 + 0.1 * rng.standard_normal(states.size)
    heavy = rng.random(states.size) < 0.12
    left_tail = -rng.pareto(1.5, states.size) * heavy
    mix = np.where(
        actions > 0.5,
        0.94 + left_tail + 0.05 * rng.standard_normal(states.size),
        r,
    )
    return np.asarray(mix)


def mc_return_dist(state: int, action: int, rng: np.random.Generator, n: int = 4000) -> FloatArray:
    """Monte-Carlo ground-truth return distribution."""
    s = np.full(n, float(state))
    a = np.full(n, float(action))
    return sample_return(s, a, rng)


def cvar(x: FloatArray, alpha: float = 0.1) -> float:
    qs = np.quantile(x, alpha)
    tail = x[x <= qs]
    return float(tail.mean()) if tail.size else float(qs)
