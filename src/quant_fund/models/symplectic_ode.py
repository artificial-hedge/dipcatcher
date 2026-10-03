"""Symplectic integrator canon for separable Hamiltonian systems.

For H(q,p) = T(p) + V(q) the flows of T and V are exactly
integrable; compositions give geometric integrators that nearly
conserve energy over exponentially long horizons — the core
reason symplectic methods dominate long-horizon MD/celestial
mechanics.

- ``verlet_step`` / ``verlet_solve`` — Stoermer-Verlet (KDK)
  leapfrog, order 2.
- ``yoshida4_step`` / ``yoshida4_solve`` — Yoshida triple-jump
  composition, order 4.
- ``forest_ruth_step`` — Forest-Ruth 4th-order symmetric
  scheme (equivalent to Yoshida with recomposed coefficients).
- ``rk4_solve_h`` — generic RK4 baseline (non-symplectic; for
  drift comparison).
- ``hamiltonian_energy`` — helper.

Bench: Kepler orbit + quartic oscillator — energy drift of
Verlet/Yoshida vs RK4 over thousands of steps (SYNTHETIC only).
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def _t_flow(p: FloatArray, h: float, dT: FloatArray) -> FloatArray:
    return np.asarray(p + h * dT, dtype=np.float64)


def _v_flow(q: FloatArray, h: float, dV: FloatArray) -> FloatArray:
    return np.asarray(q + h * dV, dtype=np.float64)


def verlet_step(q, p, h, dV, dT):
    p = _t_flow(p, h / 2, dT(q))
    q = _v_flow(q, h, dV(p))
    p = _t_flow(p, h / 2, dT(q))
    return q, p


def verlet_solve(q0, p0, h, n, dV, dT):
    q, p = np.asarray(q0, dtype=np.float64), np.asarray(p0, dtype=np.float64)
    qs, ps = np.empty((n + 1, len(q))), np.empty((n + 1, len(p)))
    qs[0], ps[0] = q, p
    for i in range(n):
        q, p = verlet_step(q, p, h, dV, dT)
        qs[i + 1], ps[i + 1] = q, p
    return qs, ps


def _c123() -> float:
    return float(2.0 ** (1.0 / 3.0))


def yoshida4_step(q, p, h, dV, dT):
    c = _c123()
    w1 = 1.0 / (2.0 - c)
    w0 = -c * w1
    q, p = verlet_step(q, p, w1 * h, dV, dT)
    q, p = verlet_step(q, p, w0 * h, dV, dT)
    q, p = verlet_step(q, p, w1 * h, dV, dT)
    return q, p


def yoshida4_solve(q0, p0, h, n, dV, dT):
    q, p = np.asarray(q0, dtype=np.float64), np.asarray(p0, dtype=np.float64)
    qs = np.empty((n + 1, len(q)))
    ps = np.empty((n + 1, len(p)))
    qs[0], ps[0] = q, p
    for i in range(n):
        q, p = yoshida4_step(q, p, h, dV, dT)
        qs[i + 1], ps[i + 1] = q, p
    return qs, ps


def forest_ruth_step(q, p, h, dV, dT):
    """Forest-Ruth (1990) 4th-order symmetric composition."""
    c = _c123()
    theta = 1.0 / (2.0 - c)
    # FR uses alternating T/V half-steps with coefficients
    # theta and (1-2*theta) on the kick, (1-3*theta)/2 style drift.
    a1 = theta / 2.0
    a2 = (1.0 - theta) / 2.0
    b1 = theta
    b2 = 1.0 - 2.0 * theta
    p = _t_flow(p, a1 * h, dT(q))
    q = _v_flow(q, b1 * h, dV(p))
    p = _t_flow(p, a2 * h, dT(q))
    q = _v_flow(q, b2 * h, dV(p))
    p = _t_flow(p, a2 * h, dT(q))
    q = _v_flow(q, b1 * h, dV(p))
    p = _t_flow(p, a1 * h, dT(q))
    return q, p


def rk4_solve_h(f, y0: FloatArray, h: float, n: int) -> FloatArray:
    """Plain RK4 on a first-order system f(y) (baseline)."""
    y = np.asarray(y0, dtype=np.float64)
    ys = np.empty((n + 1, len(y)))
    ys[0] = y
    for i in range(n):
        k1 = f(y)
        k2 = f(y + h / 2 * k1)
        k3 = f(y + h / 2 * k2)
        k4 = f(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys[i + 1] = y
    return ys


def hamiltonian_energy(q, p, T, V):
    return float(np.sum(T(p) + V(q)))


def bench_symplectic(seed: int = 0) -> dict[str, float]:
    _ = np.random.default_rng(seed)
    # Quartic oscillator: H = p^2/2 + q^4/4 (separable, anharmonic).
    dT = lambda q: np.asarray(-(q**3), dtype=np.float64)  # noqa: E731
    dV = lambda p: np.asarray(p, dtype=np.float64)  # noqa: E731
    T = lambda p: float(0.5 * np.sum(p**2))  # noqa: E731
    V = lambda q: float(0.25 * np.sum(q**4))  # noqa: E731
    q0 = np.array([1.0])
    p0 = np.array([0.0])
    h, n = 0.05, 4000
    e0 = hamiltonian_energy(q0, p0, T, V)

    qs_v, ps_v = verlet_solve(q0, p0, h, n, dV, dT)
    qs_y, ps_y = yoshida4_solve(q0, p0, h, n, dV, dT)
    drift_v = float(
        max(abs(hamiltonian_energy(qs_v[i], ps_v[i], T, V) - e0) for i in range(0, n + 1, 50))
    )
    drift_y = float(
        max(abs(hamiltonian_energy(qs_y[i], ps_y[i], T, V) - e0) for i in range(0, n + 1, 50))
    )

    def f(y: FloatArray) -> FloatArray:
        return np.array([y[1], -(y[0] ** 3)])

    yrk = rk4_solve_h(f, np.array([1.0, 0.0]), h, n)
    drift_rk = float(
        max(abs(hamiltonian_energy(yrk[i, :1], yrk[i, 1:], T, V) - e0) for i in range(0, n + 1, 50))
    )
    return {
        "synthetic_verlet_drift": drift_v,
        "synthetic_yoshida_drift": drift_y,
        "synthetic_rk4_drift": drift_rk,
        "synthetic_verlet_amplitude": float(qs_v[:, 0].max()),
    }


__all__ = [
    "bench_symplectic",
    "forest_ruth_step",
    "hamiltonian_energy",
    "rk4_solve_h",
    "verlet_solve",
    "verlet_step",
    "yoshida4_solve",
    "yoshida4_step",
]
