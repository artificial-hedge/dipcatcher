import numpy as np
import pytest

from quant_fund.models.har_rv import bench_har_rv, har_rv_fit, synth_har


def test_har_recovers_cascade() -> None:
    out = har_rv_fit(synth_har(beta=(0.3, 0.4, 0.2), seed=2))
    assert abs(out["beta_day"] - 0.3) < 0.15
    assert out["beta_week"] > 0.1
    assert out["r2"] > 0.2


def test_har_flat_path_no_fit() -> None:
    rng = np.random.default_rng(3)
    out = har_rv_fit(np.exp(rng.normal(0.0, 0.2, 900)))
    assert out["r2"] < 0.1


def test_har_multistep() -> None:
    out = har_rv_fit(synth_har(seed=4), h=5)
    assert np.isfinite(out["r2"])
    assert out["h"] == 5.0


def test_har_validation() -> None:
    with pytest.raises(ValueError):
        har_rv_fit(np.ones(100))
    with pytest.raises(ValueError):
        har_rv_fit(synth_har(t=100))
    with pytest.raises(ValueError):
        har_rv_fit(-np.abs(synth_har(seed=5)))
    with pytest.raises(ValueError):
        har_rv_fit(synth_har(seed=6), h=0)


def test_har_deterministic() -> None:
    rv = synth_har(seed=7)
    a = har_rv_fit(rv)
    b = har_rv_fit(rv)
    assert a["beta_day"] == b["beta_day"]


def test_bench_har_rv() -> None:
    out = bench_har_rv()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
