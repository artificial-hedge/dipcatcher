import numpy as np

from quant_fund.models.laplace_approx import (
    bench_laplace_approx,
    laplace_fit,
    newton_mode,
)


def test_newton_finds_mode():
    nlp = lambda w: float(np.sum((w - 2.0) ** 2))  # noqa: E731
    grad = lambda w: 2 * (w - 2.0)  # noqa: E731
    hess = lambda w: 2.0 * np.eye(w.size)  # noqa: E731
    w = newton_mode(nlp, grad, hess, np.zeros(3))
    assert np.linalg.norm(w - 2.0) < 1e-6


def test_laplace_gaussian_exact():
    # on a quadratic, Laplace is exact
    a = np.array([[2.0, 0.1], [0.1, 1.0]])
    nlp = lambda w: float(0.5 * w @ (a @ w))  # noqa: E731
    grad = lambda w: a @ w  # noqa: E731
    hess = lambda w: a  # noqa: E731
    fit = laplace_fit(nlp, grad, hess, np.array([1.0, 1.0]))
    assert np.linalg.norm(fit["mode"]) < 1e-8
    assert np.allclose(fit["cov"], np.linalg.inv(a))


def test_bench_keys():
    out = bench_laplace_approx()
    assert out["synthetic_laplace_vs_mh_err"] < 0.15
    assert out["synthetic_laplace_brier"] < 0.25
