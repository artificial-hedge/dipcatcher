"""Gillespie SSA tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.gillespie_ssa import (
    bench_gillespie,
    hill,
    mass_action,
    michaelis_menten,
    sir_network,
    ssa,
    tau_leap,
)


def test_mass_action():
    # rate x * choose(10,2) = 45
    assert mass_action(2.0, (10, 2)) == pytest.approx(90.0)
    assert mass_action(1.0, (1, 2)) == 0.0


def test_mm_hill_shapes():
    assert michaelis_menten(10, 5, 5) == pytest.approx(5.0)
    assert hill(10, 2, 2, 2) == pytest.approx(5.0)


def test_ssa_pure_birth():
    # pure birth Y: X->2X at rate k*x; E[x]=x0 e^{kt}
    V = np.array([[1]], dtype=np.int64)

    def prop(x: np.ndarray) -> np.ndarray:
        return np.array([0.5 * x[0]])

    _, st = ssa(V, prop, np.array([10]), t_end=2.0, seed=0)
    # E = 10 e^1 = 27.2; allow MC tolerance
    assert 10 < st[-1, 0] < 80


def test_sir_extinction():
    V, prop = sir_network(0.3, 1.0)  # R0 < 1
    _, st = ssa(V, prop, np.array([490, 10, 0]), t_end=50.0, seed=1)
    assert st[-1, 1] == 0


def test_tau_leap_nonnegative():
    V, prop = sir_network(1.0, 0.4)
    _, st = tau_leap(V, prop, np.array([490, 10, 0]), t_end=10.0, tau=0.1)
    assert (st >= 0).all()


def test_bench_gillespie():
    out = bench_gillespie()
    assert 0.75 < out["synthetic_attack_rate"] < 0.98
