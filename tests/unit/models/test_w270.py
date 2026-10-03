"""Wave-270 computational-physics-2 module tests."""

import numpy as np

from quant_fund.models.dmc_solver import dmc_run
from quant_fund.models.fdtd_wave import fdtd
from quant_fund.models.ising_metro import ising_sweep
from quant_fund.models.lattice_boltzmann import lbm_run
from quant_fund.models.lj_md import _energy, simulate
from quant_fund.models.pic_plasma import pic_step, solve_field


def test_lj_energy_conserved() -> None:
    rng = np.random.RandomState(0)
    pos = rng.rand(4, 2) * 6 + 2
    vel = rng.normal(0, 0.1, (4, 2))
    e0 = _energy(pos, vel)
    p, v = simulate(pos, vel, 30, dt=0.001)
    assert abs(_energy(p, v) - e0) / (abs(e0) + 1) < 0.1


def test_fdtd_cfl_stable() -> None:
    x = np.linspace(0, 1, 50)
    u0 = np.exp(-((x - 0.5) ** 2) / 0.01)
    u = fdtd(u0, 1.0, x[1] - x[0], 0.4 * (x[1] - x[0]), 40)
    assert np.isfinite(u).all()
    assert u.max() < 5


def test_lbm_mass_conserved() -> None:
    rho = lbm_run(16, 8, 30)
    assert abs(rho.sum() - 16 * 8) < 1e-3


def test_ising_orders_cold() -> None:
    rng = np.random.RandomState(0)
    s = rng.choice([-1, 1], (12, 12))
    for _ in range(15):
        s = ising_sweep(s, 0.9, rng)
    assert abs(s.mean()) > 0.6


def test_ising_disorders_hot() -> None:
    rng = np.random.RandomState(0)
    s = np.ones((12, 12))
    for _ in range(10):
        s = ising_sweep(s, 0.05, rng)
    assert abs(s.mean()) < 0.9


def test_pic_field_solves() -> None:
    rho = np.zeros(16)
    rho[4] = 5.0
    e = solve_field(rho, 0.5)
    assert np.isfinite(e).all()
    x, v = pic_step(np.array([1.0]), np.array([0.0]), e, 0.5, 0.05)
    assert np.isfinite(x).all() and np.isfinite(v).all()


def test_dmc_converges() -> None:
    rng = np.random.RandomState(0)
    e = dmc_run(rng.normal(0, 2, 200), rng, steps=100)
    assert 0.2 < e < 1.2
