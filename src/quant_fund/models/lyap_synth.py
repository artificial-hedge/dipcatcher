"""Control-Lyapunov-function synthesis (wave 279) (SYNTHETIC).

System x'' = u with V = 0.5(x1^2 + x2^2). The Sontag/CLF control
u = -x1 - c x2 renders V-dot = -c x2^2 <= 0; driving the system with the
same law produces monotone V decay. An antisymmetric (wrong-sign) controller
fails to decrease V.
"""

import numpy as np

_SEED = 20261231 + 762


def _run(seed: int, correct_sign: bool, steps: int = 400) -> np.ndarray:
    rng = np.random.RandomState(seed)
    x = rng.normal(0, 1.5, 2)
    vs: list[float] = []
    for _ in range(steps):
        u = (-x[0] - 2.0 * x[1]) if correct_sign else (x[0] + 2.0 * x[1])
        x = x + 0.01 * np.array([x[1], u])
        vs.append(0.5 * float(x @ x))
    return np.array(vs)


def bench_lyap_synth(seed: int = _SEED) -> dict[str, float]:
    v_ok = _run(seed, correct_sign=True)
    v_bad = _run(seed, correct_sign=False)
    return {
        "synthetic_lyap_decay": float(v_ok[-1] < 0.05 * v_ok[0]),
        "synthetic_lyap_order": float(v_ok[-1] < 0.05 * v_bad[-1]),
    }
