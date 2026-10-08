"""Diffusion Monte Carlo: branching random walkers sample ground-state (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 709


def dmc_run(walkers: np.ndarray, rng: np.random.RandomState, steps: int = 200) -> float:
    """Harmonic potential V = x^2/2; branch walkers by Boltzmann weight."""
    e_t = float(np.mean(0.5 * walkers * walkers))
    dt = 0.05
    for _ in range(steps):
        walkers = walkers + rng.normal(0, np.sqrt(dt), len(walkers))
        w = np.exp(-dt * (0.5 * walkers * walkers - e_t))
        n_new = np.floor(w + rng.rand(len(walkers))).astype(int)
        walkers = np.repeat(walkers, np.clip(n_new, 0, 3))
        if len(walkers) == 0:
            walkers = rng.normal(0, 1, 200)
        e_t = 0.95 * e_t + 0.05 * float(np.mean(0.5 * walkers * walkers))
        if len(walkers) > 800:
            walkers = rng.choice(walkers, 400, replace=False)
    return float(np.mean(np.abs(walkers) ** 2))


def bench_dmc_solver(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 15
    for _ in range(trials):
        w = rng.normal(0, 2, 200)
        e = dmc_run(w, rng)
        # <x^2> for HO ground state = 0.5 (exact: 0.5 * hbar omega)
        ok += float(0.2 < e < 1.2)
    return {"synthetic_dmc_ho_moment": ok / trials}
