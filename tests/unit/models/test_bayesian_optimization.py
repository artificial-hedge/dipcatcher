import numpy as np

from quant_fund.models.bayesian_optimization import (
    bayes_opt,
    bench_bayes_opt,
    expected_improvement,
    probability_improvement,
    ucb,
)


def test_ei_positive_and_zero_at_zero_sd():
    ei = expected_improvement(np.array([0.5]), np.array([0.0]), f_best=1.0)
    assert ei[0] > 0


def test_ucb_prefers_uncertainty():
    a = ucb(np.array([0.0, 0.0]), np.array([0.1, 1.0]))
    assert a[1] > a[0]


def test_pi_in_01():
    pi = probability_improvement(np.array([0.5, 2.0]), np.array([0.2, 0.2]), 1.0)
    assert np.all((pi >= 0) & (pi <= 1))
    assert pi[0] > pi[1]


def test_bayes_opt_improves_on_init():
    def f(x):
        return float((x[0] - 0.4) ** 2)

    r = bayes_opt(f, np.array([0.0]), np.array([1.0]), n_init=8, n_iter=10, seed=5)
    ys = np.asarray(r["ys"])
    assert float(r["f"]) <= ys[:8].min() + 1e-9


def test_bench_bayes_opt_runs():
    out = bench_bayes_opt(seed=541)
    assert out["synthetic_bo_best_f"] < -4.5
