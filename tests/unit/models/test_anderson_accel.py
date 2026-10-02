import numpy as np

from quant_fund.models.anderson_accel import (
    anderson_aa,
    bench_anderson,
    picard,
)


def test_picard_converges():
    g = lambda x: 0.5 * x + 1.0  # noqa: E731
    x, it = picard(g, np.array([0.0]), tol=1e-10)
    assert abs(x[0] - 2.0) < 1e-8


def test_anderson_beats_picard():
    rng = np.random.default_rng(0)
    d = 20
    m = rng.standard_normal((d, d))
    m = m.T @ m + np.eye(d)
    xs = rng.standard_normal(d)
    c = m @ xs
    lam = float(np.linalg.eigvalsh(m).max())
    eps = 1.9 / lam

    def g(x):
        return x - eps * (m @ x - c)

    x0 = np.zeros(d)
    _, n_pic = picard(g, x0, tol=1e-8, it=6000)
    xa, n_aa = anderson_aa(g, x0, m=8, tol=1e-8)
    assert n_aa < n_pic
    assert np.linalg.norm(xa - xs) < 1e-6


def test_bench_keys():
    out = bench_anderson()
    assert out["synthetic_aa_iters"] < out["synthetic_picard_iters"]
