import numpy as np

from quant_fund.models.lax_wendroff import (
    advect,
    bench_lax_wendroff,
    lax_wendroff_step,
)


def test_lw_translation():
    n = 128
    x = np.linspace(0, 1, n, endpoint=False)
    u0 = np.sin(2 * np.pi * x)
    sigma = 0.5
    # one full wrap: a*t = 1 → steps = n/sigma... but LW needs
    # a·t/dx·sigma = steps·sigma/n cells = n cells
    u = advect(u0, sigma, int(n / sigma))
    # after one period the profile returns
    err = np.abs(u - u0).max()
    assert err < 0.15


def test_lw_beats_upwind_peak():
    out = bench_lax_wendroff(seed=2)
    assert out["synthetic_lw_peak"] > out["synthetic_upwind_peak"]
    assert out["synthetic_err_ratio"] > 1.0
    assert out["synthetic_lw_mass_err"] < 1e-12


def test_cfl_violation_blows_up():
    n = 64
    u0 = np.random.default_rng(0).standard_normal(n)
    u = advect(u0, 1.4, 60, "lw")
    assert np.abs(u).max() > 10.0


def test_lw_step_conserves_mass():
    u0 = np.random.default_rng(3).standard_normal(50)
    u = lax_wendroff_step(u0, 0.4)
    assert abs(u.sum() - u0.sum()) < 1e-12
