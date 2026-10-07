"""Taylor rule + determinacy: monetary response to inflation gap (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 754


def taylor_sim(phi_pi: float, phi_y: float, steps: int, shock: float) -> np.ndarray:
    """π_{t+1} = π_t + κ x_t; x_{t+1} = x_t - σ(i_t - π_t); i = φπ·π + φy·x."""
    pi = shock
    x = 0.0
    out = np.zeros(steps)
    for t in range(steps):
        i = phi_pi * pi + phi_y * x
        x_new = x - 0.2 * (i - pi)
        pi = pi + 0.1 * x
        x = x_new
        out[t] = pi
    return out


def bench_taylor_rule(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    del rng
    det = taylor_sim(1.5, 0.5, 200, 0.02)
    indet = taylor_sim(0.5, 0.1, 200, 0.02)
    ok = float(abs(det[-1]) < 0.005 and np.isfinite(indet).all())
    return {"synthetic_taylor_determinate": ok}
