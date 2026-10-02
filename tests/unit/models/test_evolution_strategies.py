import numpy as np

from quant_fund.models.evolution_strategies import (
    bench_evolution_strategies,
    es_mu_lambda,
    es_one_plus_one,
)


def test_mu_lambda_sphere():
    f = lambda x: float(np.sum(x**2))  # noqa: E731
    r = es_mu_lambda(f, np.full(5, 2.0), sigma=1.0, it=150, seed=0)
    assert float(np.asarray(r["f"])) < 1.0


def test_one_plus_one_improves():
    f = lambda x: float(np.sum((x - 1.0) ** 2))  # noqa: E731
    r = es_one_plus_one(f, np.zeros(4), sigma=0.4, it=1500, seed=1)
    assert float(np.asarray(r["f"])) < f(np.zeros(4))


def test_bench_evolution_strategies():
    out = bench_evolution_strategies(seed=563)
    assert out["synthetic_es_ellipsoid_f"] < out["synthetic_es_baseline_f"]
    assert out["synthetic_es11_sphere_f"] < 0.1
