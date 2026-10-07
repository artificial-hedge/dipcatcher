"""Generalized Advantage Estimation (Schulman et al., 2015) — the (SYNTHETIC)
(γ, λ) exponential-weighted TD residual sum interpolating between
one-step TD (λ=0) and Monte-Carlo return (λ=1).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def gae(
    rewards: FloatArray,
    values: FloatArray,
    bootstrap_v: float,
    dones: FloatArray | None = None,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[FloatArray, FloatArray]:
    """Compute GAE advantages and λ-returns.

    rewards, values: length-T arrays (values = V(s_0..s_{T-1})).
    bootstrap_v: V(s_T). dones: optional terminal mask per step.
    Returns (advantages, returns) where returns = adv + values.
    """
    T = len(rewards)
    vnext = np.append(values[1:], bootstrap_v)
    if dones is not None:
        vnext = np.where(dones > 0, 0.0, vnext)
    deltas = rewards + gamma * vnext - values
    adv = np.zeros(T)
    acc = 0.0
    for t in range(T - 1, -1, -1):
        acc = deltas[t] + gamma * lam * acc
        if dones is not None and dones[t] > 0:
            acc = deltas[t]
        adv[t] = acc
    return adv, adv + values


def bench_gae(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: λ=0 recovers TD residuals; λ=1 recovers MC return;
    closed form on constant rewards."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    T = 50
    rewards = rng.normal(0.3, 0.1, T)
    values = rng.normal(0.5, 0.2, T)
    adv0, ret0 = gae(rewards, values, 0.4, gamma=0.9, lam=0.0)
    delta = rewards + 0.9 * np.append(values[1:], 0.4) - values
    out["synthetic_gae_lam0_err"] = float(np.abs(adv0 - delta).max())
    adv1, ret1 = gae(rewards, values, 0.0, gamma=0.9, lam=1.0)
    # λ=1: adv_t = Σ γ^{k} r_{t+k} − V_t (bootstrap 0)
    mc = np.array([sum(0.9**k * rewards[t + k] for k in range(T - t)) for t in range(T)])
    out["synthetic_gae_lam1_err"] = float(np.abs(adv1 - (mc - values)).max())
    # constant stream closed form: adv = δ·(1−(γλ)^T)/(1−γλ)
    T2 = 20
    adv_c, _ = gae(np.full(T2, 0.5), np.full(T2, 0.1), 0.1, gamma=0.9, lam=0.95)
    delta_c = 0.5 + 0.9 * 0.1 - 0.1
    expect = delta_c * (1 - (0.9 * 0.95) ** (T2)) / (1 - 0.9 * 0.95)
    out["synthetic_gae_const_err"] = float(abs(adv_c[0] - expect))
    out["synthetic_gae_dones_reset"] = float(
        abs(
            gae(np.ones(4), np.zeros(4), 9.0, dones=np.array([0.0, 1.0, 0, 0]), gamma=0.9, lam=1.0)[
                0
            ][1]
            - 1.0
        )
    )
    return out


if __name__ == "__main__":
    print(bench_gae())
