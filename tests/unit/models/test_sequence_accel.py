import numpy as np

from quant_fund.models.sequence_accel import (
    aitken_iterate,
    bench_sequence_accel,
    richardson,
    wynn_epsilon,
)


def test_aitken_on_geometric():
    s = 1.0 - 0.5 ** np.arange(1, 20)
    a = aitken_iterate(s)
    assert abs(a[-1] - 1.0) < abs(s[-1] - 1.0) / 10


def test_wynn_on_leibniz():
    n = 30
    terms = 4.0 * (-1.0) ** np.arange(n) / (2 * np.arange(n) + 1)
    s = np.cumsum(terms)
    est = wynn_epsilon(s)
    assert abs(est - np.pi) < 1e-6


def test_richardson_exact_poly():
    # f(h) = a + b h^2 — extrapolate exactly
    hs = np.array([0.5, 0.25, 0.125])
    fh = 3.0 + 2.0 * hs**2
    est = richardson(fh, hs, p=2)
    assert abs(est - 3.0) < 1e-12


def test_bench_keys():
    out = bench_sequence_accel()
    assert out["synthetic_aitken_err"] < out["synthetic_leibniz_raw_err"] / 100
    assert out["synthetic_wynn_err"] < 1e-6
    assert out["synthetic_richardson_err"] < out["synthetic_trap_err"] / 100
