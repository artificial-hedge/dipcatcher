"""Gain-scheduled controller: operating-point-varying PI (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 718


def gsc_run(steps: int = 300) -> np.ndarray:
    """Plant gain varies with setpoint; controller gain scheduled on x."""
    x = 0.0
    integ = 0.0
    dt = 0.02
    hist = []
    setpoints = np.concatenate([np.ones(150) * 1.0, np.ones(150) * 3.0])
    for t in range(min(steps, len(setpoints))):
        sp = setpoints[t]
        plant_gain = 0.5 + 0.5 * abs(x)
        kp = 1.0 / max(plant_gain, 0.2)
        err = sp - x
        integ += err * dt
        u = kp * err + 1.0 * integ
        x += dt * (-x + plant_gain * u)
        hist.append(x)
    return np.array(hist)


def bench_gain_schedule(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    del rng
    h = gsc_run()
    ok = float(abs(h[149] - 1.0) < 0.15 and abs(h[-1] - 3.0) < 0.3)
    return {"synthetic_gsc_tracks": ok}
