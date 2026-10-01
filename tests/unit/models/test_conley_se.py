import numpy as np
import pytest

from quant_fund.models.conley_se import (
    bench_conley_se,
    conley_vcov,
    synth_conley,
)


def test_conley_inflates_under_spatial_corr() -> None:
    d = synth_conley(rho_d=30.0, seed=0)
    x = np.asarray(d["x"])
    xd = np.column_stack([np.ones(x.shape[0]), x])
    b, *_ = np.linalg.lstsq(xd, np.asarray(d["y"]), rcond=None)
    e = np.asarray(d["y"]) - xd @ b
    vc = conley_vcov(xd, e, np.asarray(d["coord"]), cutoff=60.0)
    se_ols = float(np.sqrt((e @ e) / (e.size - 2) * np.linalg.inv(xd.T @ xd)[1, 1]))
    assert float(np.sqrt(vc[1, 1])) > 1.5 * se_ols


def test_conley_near_ols_when_iid() -> None:
    d = synth_conley(rho_d=0.0, seed=1)
    x = np.asarray(d["x"])
    xd = np.column_stack([np.ones(x.shape[0]), x])
    b, *_ = np.linalg.lstsq(xd, np.asarray(d["y"]), rcond=None)
    e = np.asarray(d["y"]) - xd @ b
    vc = conley_vcov(xd, e, np.asarray(d["coord"]), cutoff=60.0)
    se_ols = float(np.sqrt((e @ e) / (e.size - 2) * np.linalg.inv(xd.T @ xd)[1, 1]))
    assert float(np.sqrt(vc[1, 1])) < 2.5 * se_ols


def test_vcov_shape() -> None:
    d = synth_conley(seed=2)
    x = np.asarray(d["x"])
    xd = np.column_stack([np.ones(x.shape[0]), x])
    e = np.asarray(d["y"]) - xd @ np.linalg.lstsq(xd, np.asarray(d["y"]), rcond=None)[0]
    vc = conley_vcov(xd, e, np.asarray(d["coord"]), cutoff=40.0)
    assert vc.shape == (2, 2)


def test_2d_coord() -> None:
    rng = np.random.default_rng(3)
    n = 100
    c2 = rng.uniform(0, 10, (n, 2))
    x = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
    e = rng.normal(0, 1, n)
    vc = conley_vcov(x, e, c2, cutoff=3.0)
    assert vc.shape == (2, 2)


def test_validation() -> None:
    rng = np.random.default_rng(0)
    n = 60
    x = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
    e = rng.normal(0, 1, n)
    c = np.arange(n, dtype=float)
    with pytest.raises(ValueError):
        conley_vcov(x, e, c, cutoff=-1.0)
    with pytest.raises(ValueError):
        conley_vcov(x, e[:30], c, cutoff=10.0)
    with pytest.raises(ValueError):
        conley_vcov(x, e * np.nan, c, cutoff=10.0)


def test_bench() -> None:
    out = bench_conley_se()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
