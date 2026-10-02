import numpy as np

from quant_fund.models.symplectic_ode import (
    bench_symplectic,
    forest_ruth_step,
    hamiltonian_energy,
    verlet_solve,
    yoshida4_solve,
)


def _osc():
    dT = lambda q: -q  # noqa: E731
    dV = lambda p: p  # noqa: E731
    T = lambda p: float(0.5 * np.sum(p**2))  # noqa: E731
    V = lambda q: float(0.5 * np.sum(q**2))  # noqa: E731
    return dT, dV, T, V


def test_verlet_energy_bounded():
    dT, dV, T, V = _osc()
    qs, ps = verlet_solve(np.array([1.0]), np.array([0.0]), 0.1, 2000, dV, dT)
    e0 = hamiltonian_energy(qs[0], ps[0], T, V)
    drift = max(abs(hamiltonian_energy(qs[i], ps[i], T, V) - e0) for i in range(0, 2000, 100))
    assert drift < 0.01


def test_yoshida_higher_order():
    dT, dV, T, V = _osc()
    q0, p0 = np.array([1.0]), np.array([0.0])
    qs, ps = yoshida4_solve(q0, p0, 0.2, 200, dV, dT)
    e0 = hamiltonian_energy(q0, p0, T, V)
    drift = abs(hamiltonian_energy(qs[-1], ps[-1], T, V) - e0)
    assert drift < 1e-4


def test_forest_ruth_runs():
    dT, dV, _, _ = _osc()
    q, p = np.array([1.0]), np.array([0.0])
    for _ in range(10):
        q, p = forest_ruth_step(q, p, 0.05, dV, dT)
    assert np.all(np.isfinite(q)) and np.all(np.isfinite(p))


def test_bench_keys():
    out = bench_symplectic()
    assert out["synthetic_verlet_drift"] < 0.01
    assert out["synthetic_yoshida_drift"] < out["synthetic_verlet_drift"]
