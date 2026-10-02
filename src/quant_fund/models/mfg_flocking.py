"""Cucker-Smale flocking mean-field game (kinetic alignment).

Particles: dx_i = v_i dt, dv_i = (k/N) sum_j phi(|x_j - x_i|)(v_j - v_i) dt
with phi(r) = 1/(1 + r^2). The mean-field interaction drives velocity
variance to zero; bench measures alignment vs an interaction-free swarm.
"""

import numpy as np

from quant_fund.models._mfg_synth import POT_N


def _simulate(
    seed: int, n: int = 120, steps: int = 240, k: float = 1.0, dt: float = 0.02, couple: bool = True
) -> float:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 5.0, n)
    v = rng.normal(0.0, 1.5, n)
    for _ in range(steps):
        if couple:
            dist = x[:, None] - x[None, :]
            w = 1.0 / (1.0 + dist**2)
            dv = (k / n) * (w * (v[None, :] - v[:, None])).sum(axis=1)
            v = v + dv * dt
        x = x + v * dt
    return float(np.var(v))


def bench_mfg_flocking(seed: int = 4203) -> dict[str, float]:
    var_coupled = _simulate(seed)
    var_free = _simulate(seed, couple=False)
    return {
        "synthetic_flock_var": var_coupled,
        "synthetic_flock_free_var": var_free,
        "synthetic_flock_align_ratio": var_coupled / max(var_free, 1e-12),
        "synthetic_flock_n": float(POT_N),
    }
