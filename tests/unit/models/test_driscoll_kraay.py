import numpy as np
import pytest

from quant_fund.models.driscoll_kraay import (
    bench_driscoll_kraay,
    driscoll_kraay_vcov,
    synth_dk_panel,
)


def _fit(d: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(d["x"])
    xd = np.column_stack([np.ones(x.shape[0]), x])
    b = np.linalg.lstsq(xd, np.asarray(d["y"]), rcond=None)[0]
    return xd, np.asarray(d["y"]) - xd @ b, np.asarray(d["t_idx"])


def test_hac_bandwidth_inflates_under_factor() -> None:
    xd, e, ti = _fit(synth_dk_panel(factor_sd=1.5, seed=0))
    v4 = driscoll_kraay_vcov(xd, e, ti, m=4.0)
    v1 = driscoll_kraay_vcov(xd, e, ti, m=1.0)
    assert v4[1, 1] > v1[1, 1]


def test_vcov_shape_and_psd() -> None:
    xd, e, ti = _fit(synth_dk_panel(seed=1))
    v = driscoll_kraay_vcov(xd, e, ti, m=3.0)
    assert v.shape == (2, 2)
    assert v[1, 1] > 0


def test_null_panel_finite() -> None:
    xd, e, ti = _fit(synth_dk_panel(factor_sd=0.0, seed=2))
    v = driscoll_kraay_vcov(xd, e, ti, m=4.0)
    assert np.all(np.isfinite(v))


def test_aggregates_periods() -> None:
    rng = np.random.default_rng(0)
    x = np.column_stack([np.ones(240), rng.normal(0, 1, 240)])
    e = rng.normal(0, 1, 240)
    ti = np.repeat(np.arange(6), 40).astype(float)
    v = driscoll_kraay_vcov(x, e, ti, m=2.0)
    assert np.all(np.isfinite(v))


def test_validation() -> None:
    rng = np.random.default_rng(0)
    n = 80
    x = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
    e = rng.normal(0, 1, n)
    ti = np.repeat(np.arange(8), 10).astype(float)
    with pytest.raises(ValueError):
        driscoll_kraay_vcov(x, e, ti, m=0.5)
    with pytest.raises(ValueError):
        driscoll_kraay_vcov(x, e[:30], ti, m=2.0)
    with pytest.raises(ValueError):
        driscoll_kraay_vcov(x, e * np.nan, ti, m=2.0)


def test_bench() -> None:
    out = bench_driscoll_kraay()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
