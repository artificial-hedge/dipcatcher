"""RBC model: log-linearized capital dynamics + TFP AR(1) (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 752


def rbc_run(steps: int, rho_a: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """k_{t+1} = gk*k_t + ga*a_t; a_{t+1} = rho a_t + eps."""
    rng = np.random.RandomState(seed)
    gk, ga = 0.95, 0.05
    a = np.zeros(steps)
    k = np.zeros(steps)
    for t in range(1, steps):
        a[t] = rho_a * a[t - 1] + rng.normal(0, 0.01)
        k[t] = gk * k[t - 1] + ga * a[t]
    return k, a


def bench_rbc_sim(seed: int = _SEED) -> dict[str, float]:
    k, a = rbc_run(800, 0.95, seed)
    # capital comoves with *lagged* TFP (accumulation lag): max corr over lags
    best = max(
        float(np.corrcoef(k[100 + lag :], a[100 : len(a) - lag])[0, 1]) for lag in range(1, 12)
    )
    stat = float(np.abs(k[-100:]).max() < 1.0)
    return {"synthetic_rbc_comove": float(best > 0.8), "synthetic_rbc_stable": stat}
