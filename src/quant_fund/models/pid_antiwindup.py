"""PID with anti-windup clamping vs naive integrator."""

import numpy as np

_SEED = 20261231 + 716


def pid_run(
    plant_a: float,
    setpoint: float,
    steps: int,
    kp: float,
    ki: float,
    u_max: float,
    antiwindup: bool,
) -> np.ndarray:
    """First-order plant x' = a x + u; naive vs clamped integrator."""
    x = 0.0
    integ = 0.0
    dt = 0.01
    out = np.zeros(steps)
    for t in range(steps):
        err = setpoint - x
        integ += err * dt
        u = kp * err + ki * integ
        u_c = float(np.clip(u, -u_max, u_max))
        if antiwindup and u != u_c:
            integ -= err * dt  # back-calculate: undo windup
        x += dt * (plant_a * x + u_c)
        out[t] = x
    return out


def bench_pid_antiwindup(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    del rng  # deterministic plant/setpoint sweep
    naive = pid_run(-0.5, 1.0, 400, 5.0, 8.0, 1.0, antiwindup=False)
    aw = pid_run(-0.5, 1.0, 400, 5.0, 8.0, 1.0, antiwindup=True)
    over_n = naive.max() - 1.0
    over_a = aw.max() - 1.0
    # antiwindup overshoots less (or equal); both eventually track
    ok = float(over_a <= over_n + 1e-9 and abs(aw[-1] - 1.0) < 0.1)
    return {"synthetic_aw_less_overshoot": ok}
