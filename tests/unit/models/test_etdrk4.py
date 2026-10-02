import numpy as np

from quant_fund.models.etdrk4 import bench_etdrk4, etdrk4, phi1


def test_phi1():
    z = np.array([0.0, 1.0, -1.0])
    p = phi1(z)
    assert abs(p[0] - 1.0) < 1e-6
    assert abs(p[1] - (np.e - 1)) < 1e-10
    assert abs(p[2] - (1 - np.exp(-1))) < 1e-10


def test_linear_exact():
    L = np.array([-3.0])

    def nlin(u):
        return np.zeros_like(u)

    t = np.linspace(0, 1, 21)
    y = etdrk4(L, nlin, np.array([1.0]), t)
    assert abs(y[-1, 0] - np.exp(-3)) < 1e-10


def test_bench():
    out = bench_etdrk4(seed=5)
    assert 3.5 < out["synthetic_etdrk4_order"] < 4.5
    assert out["synthetic_etdrk4_err"] < 1e-8
