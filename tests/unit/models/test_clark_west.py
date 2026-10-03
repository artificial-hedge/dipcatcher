import numpy as np
import pytest

from quant_fund.models.clark_west import (
    bench_clark_west,
    clark_west,
    synth_forecasts,
)


def test_cw_rejects_when_predictable() -> None:
    d = synth_forecasts(effect=0.5, seed=2)
    out = clark_west(d["y"], d["pred1"], d["pred2"])
    assert out["cw_t"] > 1.28
    assert out["p_one_sided"] < 0.1


def test_cw_null_no_strong_reject() -> None:
    d = synth_forecasts(effect=0.0, seed=3)
    out = clark_west(d["y"], d["pred1"], d["pred2"])
    assert abs(out["cw_t"]) < 3.5


def test_cw_mspe_fields() -> None:
    d = synth_forecasts(effect=0.4, seed=4)
    out = clark_west(d["y"], d["pred1"], d["pred2"])
    assert out["mspe1"] > 0
    assert out["mspe2"] > 0
    assert np.isfinite(out["mspe_ratio"])


def test_cw_validation() -> None:
    d = synth_forecasts(seed=5)
    with pytest.raises(ValueError):
        clark_west(d["y"], d["pred1"], d["pred1"])
    with pytest.raises(ValueError):
        clark_west(d["y"][:30], d["pred1"][:30], d["pred2"][:30])
    with pytest.raises(ValueError):
        clark_west(d["y"], d["pred1"][:-1], d["pred2"])


def test_cw_deterministic() -> None:
    d = synth_forecasts(effect=0.4, seed=6)
    a = clark_west(d["y"], d["pred1"], d["pred2"])
    b = clark_west(d["y"], d["pred1"], d["pred2"])
    assert a["cw_t"] == b["cw_t"]


def test_bench_clark_west() -> None:
    out = bench_clark_west()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
