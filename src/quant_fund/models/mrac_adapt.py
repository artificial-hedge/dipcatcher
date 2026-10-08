"""MIT-rule model-reference adaptive control (wave 279) (SYNTHETIC).

Plant: x' = a x + b u, unknown b. Reference model: xm' = am xm + bm r.
Adaptive law dk/dt = -gamma * e * xm with u = k r + kx x — the adaptive gain
learns to cancel the plant/model mismatch while a frozen wrong gain cannot.
"""

import numpy as np

_SEED = 20261231 + 760


def _run(seed: int, adaptive: bool, steps: int = 800) -> float:
    rng = np.random.RandomState(seed)
    a_p, b_p = -1.0, 2.0
    am, bm = -2.0, 2.0
    x, xm = 0.0, 0.0
    k, kx = 1.0, 0.0
    errs: list[float] = []
    for t in range(steps):
        r = float(np.sin(0.02 * t))
        u = k * r + kx * x
        x += 0.01 * (a_p * x + b_p * u) + 0.0001 * rng.normal()
        xm += 0.01 * (am * xm + bm * r)
        e = x - xm
        if adaptive:
            k += 0.01 * (-0.5 * e * r)
            kx += 0.01 * (-0.5 * e * x)
        errs.append(abs(e))
    return float(np.mean(errs[-200:]))


def bench_mrac_adapt(seed: int = _SEED) -> dict[str, float]:
    err_frozen = _run(seed, adaptive=False)
    err_adapt = _run(seed, adaptive=True)
    return {
        "synthetic_mrac_better": float(err_adapt < 0.6 * err_frozen),
        "synthetic_mrac_track": float(err_adapt < 0.2),
    }
