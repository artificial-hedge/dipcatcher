import numpy as np

from quant_fund.models.tensor_power import (
    bench_tensor_power,
    tensor_deflation,
    tensor_power_iteration,
    whiten,
)


def test_power_recovers_component():
    v = np.array([1.0, 0.0, 0.0])
    t = 2.0 * np.einsum("i,j,k->ijk", v, v, v)
    hat, lam = tensor_power_iteration(t, n_init=5, seed=0)
    assert abs(hat[0]) > 0.99
    assert abs(lam - 2.0) < 1e-6


def test_deflation_two_components():
    d = 4
    e1 = np.eye(d)[0]
    e2 = np.eye(d)[1]
    t = 3.0 * np.einsum("i,j,k->ijk", e1, e1, e1) + 1.5 * np.einsum("i,j,k->ijk", e2, e2, e2)
    vecs, lams = tensor_deflation(t, 2, n_init=10, seed=0)
    assert abs(vecs[0, 0]) > 0.95
    assert abs(lams[0] - 3.0) < 1e-6


def test_whiten_identity():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((500, 5))
    w, xw = whiten(x)
    cov = xw.T @ xw / 500
    assert np.allclose(cov, np.eye(cov.shape[0]), atol=0.15)


def test_bench_keys():
    out = bench_tensor_power()
    assert out["synthetic_tp_cos_min"] > 0.9
