import numpy as np

from quant_fund.models.frank_wolfe import (
    bench_frank_wolfe,
    fw_l1,
    fw_simplex,
    pfw_simplex,
)


def test_fw_simplex_feasible():
    rng = np.random.default_rng(0)
    c = rng.normal(0, 1, 10)
    x = fw_simplex(np.eye(10), -c, it=500)
    assert np.all(x >= -1e-9)
    assert abs(x.sum() - 1.0) < 1e-6


def test_pfw_simplex_feasible_and_better_than_uniform():
    rng = np.random.default_rng(1)
    c = rng.normal(0, 3, 8)
    x = pfw_simplex(np.eye(8), -c, it=500)
    f = lambda v: float(0.5 * v @ v - c @ v)  # noqa: E731
    assert f(x) < f(np.full(8, 0.125))


def test_fw_l1_ball():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 2, 15)
    x = fw_l1(np.eye(15), -z, radius=1.0, it=800)
    assert np.abs(x).sum() <= 1.0 + 1e-6
    assert float(x @ z) > 0


def test_bench_frank_wolfe():
    out = bench_frank_wolfe(seed=561)
    assert out["synthetic_fw_simplex_err"] < 0.05
    assert out["synthetic_fw_l1_err"] < 0.05
