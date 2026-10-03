"""V-trace (Espeholt et al., IMPALA, 2018) — off-policy value
correction with clipped importance sampling. Recovers the n-step
on-policy return when behaviour = target policy; corrects bias when
they differ, with c_i clipping the trace and ρ̄ clipping pointwise
corrections.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def vtrace(
    rewards: FloatArray,
    values: FloatArray,
    bootstrap_v: float,
    rhos: FloatArray,
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> FloatArray:
    """V-trace targets v_s for an n-step window.

    rhos_t = π(a_t|s_t)/μ(a_t|s_t). Returns vs_0..vs_{n-1}
    """
    n = len(rewards)
    vnext = np.append(values[1:], bootstrap_v)
    clipped_rhos = np.minimum(rhos, rho_bar)
    clipped_cs = np.minimum(rhos, c_bar)
    deltas = clipped_rhos * (rewards + gamma * vnext - values)
    vs = np.zeros(n)
    acc = 0.0
    for t in range(n - 1, -1, -1):
        acc = deltas[t] + gamma * clipped_cs[t] * acc
        vs[t] = values[t] + acc
    return vs


def bench_vtrace(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: on-policy V-trace = n-step return; off-policy ρ>1
    correction bounded by ρ̄; low-ρ drift toward V."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n = 8
    rewards = np.full(n, 0.4)
    values = rng.normal(0.0, 0.05, n)
    bootstrap = 0.7
    # on-policy: vs_0 = Σ γ^k r + γ^n bootstrap
    vs = vtrace(rewards, values, bootstrap, np.ones(n), gamma=0.9)
    expect = sum(0.9**k * 0.4 for k in range(n)) + 0.9**n * bootstrap
    out["synthetic_vtrace_onpolicy_err"] = abs(float(vs[0]) - expect)
    # ρ>1 with ρ̄=1: pointwise correction clipped to on-policy value
    vs2 = vtrace(rewards, values, bootstrap, np.full(n, 5.0), gamma=0.9, rho_bar=1.0, c_bar=1.0)
    out["synthetic_vtrace_clip_eq"] = float(np.abs(vs - vs2).max())
    # ρ>1 with ρ̄=∞, unclipped deltas → larger corrections
    vs3 = vtrace(
        rewards - values, np.zeros(n), 0.0, np.full(n, 2.0), gamma=0.9, rho_bar=10.0, c_bar=10.0
    )
    vs3b = vtrace(rewards - values, np.zeros(n), 0.0, np.ones(n), gamma=0.9)
    out["synthetic_vtrace_offpolicy_ratio"] = float(abs(vs3[0]) / (abs(vs3b[0]) + 1e-12))
    # c_i low → trace truncated: with c=0, vs = V + clipped delta only
    vs4 = vtrace(rewards, np.zeros(n), bootstrap, np.ones(n), gamma=0.9, c_bar=0.0)
    out["synthetic_vtrace_c0_err"] = float(
        np.abs(vs4 - (rewards + 0.9 * np.append(np.zeros(n - 1), bootstrap))).max()
    )
    out["synthetic_vtrace_ok"] = float(
        out["synthetic_vtrace_onpolicy_err"] < 1e-10
        and out["synthetic_vtrace_clip_eq"] < 1e-12
        and out["synthetic_vtrace_c0_err"] < 1e-12
    )
    return out


if __name__ == "__main__":
    print(bench_vtrace())
