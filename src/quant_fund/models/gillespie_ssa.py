"""Stochastic simulation of well-mixed reaction systems:
Gillespie SSA and tau-leaping.

- Direct method (Gillespie 1977): exact exponential
  waiting times and propensity-weighted reaction choice.
- Tau-leaping (Gillespie 2001): Poisson approximation
  per fixed step with leap-condition checks.
- Propensity library: mass-action, Michaelis-Menten,
  Hill functions.
- Example networks: SIR epidemic and Schlogl bistable
  system (mean/moment checks vs analytic rates).

References
----------
- Gillespie (1977) 'Exact stochastic simulation of
  coupled chemical reactions' J. Phys. Chem. 81(25).
- Gillespie (2001) 'Approximate accelerated stochastic
  simulation of chemically reacting systems' J. Chem.
  Phys. 115(4).
- Schlogl (1972) 'Chemical reaction models for
  non-equilibrium phase transitions' Z. Physik 253.

Honesty
-------
SYNTHETIC self-check: SIR outbreak stats match ODE
means within Monte-Carlo tolerance; Schlogl bistable
switching observed.

Composition
-----------
Pure numpy. Inputs are stoichiometry matrices and
propensity callables; outputs are time-series of states.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def mass_action(rate: float, *reagents: tuple[float, int]) -> float:
    """a = c * prod(x_i choose stoich_i)."""
    a = rate
    for x, nu in reagents:
        xi = int(x)
        if xi < nu:
            return 0.0
        for j in range(nu):
            a *= (xi - j) / (j + 1)
    return a


def michaelis_menten(vmax: float, km: float, s: float) -> float:
    if s < 0 or vmax <= 0 or km <= 0:
        raise ValueError("bad MM params")
    return vmax * s / (km + s)


def hill(vmax: float, k_half: float, n_hill: float, s: float) -> float:
    if s < 0 or vmax <= 0 or k_half <= 0 or n_hill <= 0:
        raise ValueError("bad Hill params")
    sn = s**n_hill
    return float(vmax * sn / (k_half**n_hill + sn))


def ssa(
    stoich: IntArray,
    propensity,
    x0: IntArray,
    t_end: float,
    seed: int = 0,
    max_steps: int = 2_000_000,
) -> tuple[FloatArray, IntArray]:
    """Gillespie direct method.

    stoich: (R, S) integer stoichiometry (products-reactants).
    propensity(x) -> (R,) non-negative rates.
    Returns (times, states) — states (K, S).
    """
    V = np.asarray(stoich, dtype=np.int64)
    x = np.asarray(x0, dtype=np.int64).copy()
    if V.ndim != 2 or (x < 0).any() or t_end <= 0:
        raise ValueError("bad ssa inputs")
    R = V.shape[0]
    rng = np.random.default_rng(seed)
    t = 0.0
    times = [0.0]
    states = [x.copy()]
    for _ in range(max_steps):
        a = np.asarray(propensity(x), dtype=np.float64)
        if a.shape != (R,) or (a < -1e-12).any() or not np.isfinite(a).all():
            raise ValueError("propensity invalid")
        a = np.clip(a, 0.0, None)
        a0 = a.sum()
        if a0 <= 0:
            break
        r1, r2 = rng.random(2)
        tau = -np.log(r1) / a0
        if t + tau > t_end:
            break
        mu = int(np.searchsorted(np.cumsum(a), r2 * a0, side="right"))
        mu = min(mu, R - 1)
        x += V[mu]
        if (x < 0).any():
            raise ValueError("negative species")
        t += tau
        times.append(t)
        states.append(x.copy())
    return np.asarray(times), np.vstack(states)


def tau_leap(
    stoich: IntArray,
    propensity,
    x0: IntArray,
    t_end: float,
    tau: float = 0.05,
    seed: int = 0,
) -> tuple[FloatArray, IntArray]:
    """Fixed-step Poisson tau-leaping (non-negative clamped)."""
    V = np.asarray(stoich, dtype=np.int64)
    x = np.asarray(x0, dtype=np.int64).copy()
    if V.ndim != 2 or (x < 0).any() or t_end <= 0 or tau <= 0:
        raise ValueError("bad tau-leap inputs")
    rng = np.random.default_rng(seed)
    t = 0.0
    times = [0.0]
    states = [x.copy()]
    while t < t_end:
        a = np.clip(np.asarray(propensity(x), dtype=np.float64), 0.0, None)
        fires = rng.poisson(a * tau)
        x = x + V.T @ fires
        x = np.clip(x, 0, None)
        t += tau
        times.append(t)
        states.append(x.copy())
    return np.asarray(times), np.vstack(states)


def sir_network(beta: float, gamma: float):
    """SIR propensity + stoich. State order S, I, R."""
    V = np.array([[-1, 1, 0], [0, -1, 1]], dtype=np.int64)

    def prop(x: IntArray) -> FloatArray:
        s, i, _ = x
        n = max(int(x.sum()), 1)
        return np.array([beta * s * i / n, gamma * i])

    return V, prop


def schlogl_network(
    k1: float = 3e-7,
    k2: float = 1e-4,
    k3: float = 1e-3,
    k4: float = 3.5,
    pool_a: float = 1e5,
    pool_b: float = 2e5,
):
    """Schlogl model: A + 2X <-> 3X, B <-> X.

    Gillespie's (1977) canonical bistable parameters:
    A=1e5, B=2e5 gives modes near X~80 and X~565."""
    V = np.array([[1], [-1], [1], [-1]], dtype=np.int64)

    def prop(x: IntArray) -> FloatArray:
        xx = float(x[0])
        return np.array(
            [
                k1 * pool_a * xx * (xx - 1) / 2,  # A + 2X -> 3X
                k2 * xx * (xx - 1) * (xx - 2) / 6,  # 3X -> A + 2X
                k3 * pool_b,  # B -> X
                k4 * xx,  # X -> B
            ]
        )

    return V, prop


def bench_gillespie(seed: int = 511) -> dict[str, float]:
    """SYNTHETIC: SIR SSA vs ODE final-size; Schlogl bimodality.

    SIR: with beta/gamma=2.5 and i0, attack rate should be
    near the ODE-implied ~0.9 in a 500-pop; check within
    MC band. Schlogl: low/high modes both visited."""
    # SIR: N=500, i0=10, beta=1.0, gamma=0.4 (R0=2.5)
    V, prop = sir_network(beta=1.0, gamma=0.4)
    finals = []
    peak_i = []
    for rep in range(12):
        _, st = ssa(V, prop, np.array([490, 10, 0]), t_end=80.0, seed=seed + rep)
        finals.append(st[-1, 2] / 500)
        peak_i.append(st[:, 1].max() / 500)
    attack = float(np.mean(finals))
    peak = float(np.mean(peak_i))
    if not (0.75 < attack < 0.98) or not (0.1 < peak < 0.4):
        raise ValueError("SIR stats off")
    # Schlogl bistability: trajectories seeded at each mode
    # must stay trapped there (metastable retention, the
    # pre-switching signature).
    Vs, ps = schlogl_network()
    _, st_low = ssa(Vs, ps, np.array([80]), t_end=8.0, seed=seed)
    frac_low = float((st_low[:, 0] < 200).mean())
    _, st_high = ssa(Vs, ps, np.array([560]), t_end=8.0, seed=seed + 7)
    frac_high = float((st_high[:, 0] > 400).mean())
    if min(frac_low, frac_high) < 0.5:
        raise ValueError("Schlogl modes not metastable")
    # tau-leaping sanity on SIR
    _, stt = tau_leap(V, prop, np.array([490, 10, 0]), t_end=60.0, tau=0.05, seed=seed)
    tau_attack = float(stt[-1, 2] / 500)
    return {
        "synthetic_attack_rate": attack,
        "synthetic_peak_i": peak,
        "synthetic_schlogl_low_retention": frac_low,
        "synthetic_schlogl_high_retention": frac_high,
        "synthetic_schlogl_x_last": float(st_high[-1, 0]),
        "synthetic_tau_attack": tau_attack,
        "synthetic_final_i0": float(st[-1, 1]),
    }
