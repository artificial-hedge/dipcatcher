import numpy as np

from quant_fund.models.cross_entropy_method import (
    bench_cross_entropy_method,
    cem_optimize,
)


def test_cem_sphere():
    f = lambda x: float(np.sum((x - 1.0) ** 2))  # noqa: E731
    out = cem_optimize(f, 4, it=40, pop=50, seed=0)
    assert out["f"] < 0.05


def test_cem_hist_monotone():
    f = lambda x: float(np.sum(x**2))  # noqa: E731
    out = cem_optimize(f, 3, it=20, pop=40, seed=1)
    hist = np.asarray(out["hist"])
    assert np.all(np.diff(hist) <= 1e-12)


def test_bench_keys():
    out = bench_cross_entropy_method()
    assert out["synthetic_cem_sphere_f"] < 0.1
    assert out["synthetic_cem_sphere_err"] < 0.5
    assert out["synthetic_cem_rastrigin_f"] < 1.0
