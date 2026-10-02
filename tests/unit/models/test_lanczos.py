import numpy as np

from quant_fund.models.lanczos import bench_lanczos, lanczos, ritz_values


def test_lanczos_recovers_top_eig():
    rng = np.random.default_rng(0)
    n = 100
    a = rng.standard_normal((n, n))
    a = (a + a.T) / 2 + 5 * np.eye(n)
    al, be = lanczos(lambda x: a @ x, n, 40, rng, reorth=True)
    top = ritz_values(al, be, 3)
    true = np.linalg.eigvalsh(a)[-3:]
    assert np.allclose(np.sort(top), np.sort(true), atol=1e-8)


def test_tridiagonal_shape():
    rng = np.random.default_rng(1)
    a = np.diag(np.arange(1.0, 11.0)) + 0.5 * np.ones((10, 10))
    al, be = lanczos(lambda x: a @ x, 10, 5, rng)
    assert len(al) == 5 and len(be) == 4


def test_bench_reorth_beats_plain():
    out = bench_lanczos(seed=2)
    assert out["synthetic_ritz_err_reorth"] < 1e-10
    assert out["synthetic_ritz_err_reorth"] < out["synthetic_ritz_err_plain"]
