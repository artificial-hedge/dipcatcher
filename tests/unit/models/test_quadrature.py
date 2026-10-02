import numpy as np

from quant_fund.models.quadrature import (
    adaptive_simpson,
    bench_quadrature,
    clenshaw_curtis,
    gauss_hermite,
    gauss_legendre,
)


def test_gauss_legendre_exact_poly():
    x, w = gauss_legendre(-1.0, 1.0, 4)
    assert abs(w @ x**7) < 1e-12
    assert abs(w @ x**2 - 2 / 3) < 1e-12


def test_gauss_hermite_moments():
    x, w = gauss_hermite(10)
    assert abs(w @ x**0 - np.sqrt(np.pi)) < 1e-10
    assert abs(w @ x**2 - 0.5 * np.sqrt(np.pi)) < 1e-10


def test_clenshaw_curtis_sin():
    x, w = clenshaw_curtis(0.0, np.pi, 16)
    assert abs(w @ np.sin(x) - 2.0) < 1e-10


def test_adaptive_simpson_exp():
    got = adaptive_simpson(np.exp, 0.0, 1.0, tol=1e-10)
    assert abs(got - (np.e - 1.0)) < 1e-8


def test_bench_keys():
    out = bench_quadrature()
    assert out["synthetic_gl_poly_err"] < 1e-8
    assert out["synthetic_gh_moment_err"] < 1e-8
