"""Repetitive control: learns periodic disturbance rejection."""

import numpy as np

_SEED = 20261231 + 721


def rep_run(period: int, steps: int) -> np.ndarray:
    """Internal-model disturbance buffer; error decays across periods."""
    x = 0.0
    buf = np.zeros(period)
    dt = 0.05
    hist = []
    for t in range(steps):
        d = np.sin(2 * np.pi * t / period)
        u = buf[t % period]
        err = d - x
        buf[t % period] += 0.3 * err
        x += dt * (-x + u)
        hist.append(abs(err))
    return np.array(hist)


def bench_repetitive_ctrl(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    del rng
    h = rep_run(40, 400)
    early = h[:40].mean()
    late = h[-40:].mean()
    ok = float(late < early)
    return {"synthetic_rep_learns": ok}
