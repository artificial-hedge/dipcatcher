import numpy as np
import pytest

from quant_fund.metrics.variance_ratio import (
    bench_variance_ratio,
    synth_vr,
    variance_ratio,
)


def test_vr_detects_positive_ar() -> None:
    out = variance_ratio(synth_vr(kind="ar_pos", ar=0.3, seed=2), q=8)
    assert out["vr"] > 1.1
    assert out["p_heterosk"] < 0.05


def test_vr_detects_negative_ma() -> None:
    out = variance_ratio(synth_vr(kind="ma_neg", ar=0.4, seed=3), q=8)
    assert out["vr"] < 0.9


def test_vr_iid_near_one() -> None:
    out = variance_ratio(synth_vr(kind="iid", seed=4), q=8)
    assert abs(out["vr"] - 1.0) < 0.2
    assert out["p_heterosk"] > 0.05


def test_vr_one_lag_free() -> None:
    out = variance_ratio(synth_vr(kind="iid", seed=5), q=2)
    assert np.isfinite(out["z_homosk"])


def test_vr_validation() -> None:
    with pytest.raises(ValueError):
        variance_ratio(np.zeros(120))
    with pytest.raises(ValueError):
        variance_ratio(synth_vr(t=50))
    with pytest.raises(ValueError):
        variance_ratio(synth_vr(seed=6), q=1)


def test_vr_deterministic() -> None:
    r = synth_vr(kind="ar_pos", seed=7)
    a = variance_ratio(r, q=8)
    b = variance_ratio(r, q=8)
    assert a["vr"] == b["vr"]


def test_bench_variance_ratio() -> None:
    out = bench_variance_ratio()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
