import numpy as np
from scipy.linalg import expm as sexpm

from quant_fund.models.expm_pade import (
    bench_expm_pade,
    expm_action,
    expm_pade,
    zoh_discretize,
)


def test_rotation_exact():
    th = 0.9
    A = np.array([[0.0, -th], [th, 0.0]])
    E = expm_pade(A)
    assert np.abs(E - np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])).max() < 1e-12


def test_random_matches_scipy():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    assert np.abs(expm_pade(A) - sexpm(A)).max() < 1e-8


def test_scaled_path():
    A = np.array([[-80.0, 40.0], [-40.0, -80.0]])
    assert np.abs(expm_pade(A) - sexpm(A)).max() < 1e-20


def test_expm_action():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 5))
    v = rng.standard_normal(5)
    assert np.abs(expm_action(A, v) - sexpm(A) @ v).max() < 1e-8


def test_zoh():
    Ad, Bd = zoh_discretize(np.array([[-2.0]]), np.array([[3.0]]), 0.4)
    assert abs(Ad[0, 0] - np.exp(-0.8)) < 1e-12
    assert abs(Bd[0, 0] - 3.0 * (1 - np.exp(-0.8)) / 2.0) < 1e-12


def test_bench():
    out = bench_expm_pade(seed=1)
    assert out["synthetic_expm_rot_err"] < 1e-12
    assert out["synthetic_zoh_ad_err"] < 1e-12
