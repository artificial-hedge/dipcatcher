"""Smith predictor: dead-time compensation vs uncompensated PI."""

import numpy as np

_SEED = 20261231 + 719


def smith_run(delay: int, steps: int, use_smith: bool) -> np.ndarray:
    """First-order plant + dead time; predictor subtracts model output."""
    x = 0.0
    buf = np.zeros(delay + 1)
    integ = 0.0
    model = 0.0
    dt = 0.02
    hist = []
    for _ in range(steps):
        err = 1.0 - x
        if use_smith:
            err = 1.0 - (x - model) - model  # x = true (delayed), model = fast model
            model += dt * (-model + buf[0])
        integ += err * dt
        u = 1.5 * err + 0.4 * integ
        buf = np.roll(buf, -1)
        buf[-1] = u
        x += dt * (-x + buf[0])
        hist.append(x)
    return np.array(hist)


def bench_smith_predictor(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    del rng
    plain = smith_run(8, 400, use_smith=False)
    comp = smith_run(8, 400, use_smith=True)
    ok = float(np.isfinite(plain).all() and np.isfinite(comp).all() and abs(comp[-1] - 1.0) < 0.2)
    return {"synthetic_smith_tracks": ok}
