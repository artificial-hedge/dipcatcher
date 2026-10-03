import numpy as np
import pytest

from quant_fund.models.pin_model import (
    bench_pin_model,
    pin_estimate,
    synth_pin,
)


def test_pin_recovers_alpha() -> None:
    d = synth_pin(alpha=0.3, mu=30.0, seed=0)
    out = pin_estimate(d["buys"], d["sells"])
    assert abs(out["alpha"] / 0.3 - 1) < 0.4


def test_pin_recovers_mu() -> None:
    d = synth_pin(alpha=0.25, mu=40.0, seed=1)
    out = pin_estimate(d["buys"], d["sells"])
    assert abs(out["mu"] / 40.0 - 1) < 0.35


def test_pin_ordering() -> None:
    d1 = synth_pin(alpha=0.35, seed=2)
    d0 = synth_pin(alpha=0.0, seed=3)
    a = pin_estimate(d1["buys"], d1["sells"])
    b = pin_estimate(d0["buys"], d0["sells"])
    assert a["pin"] > b["pin"]


def test_pin_bounds() -> None:
    d = synth_pin(seed=4)
    out = pin_estimate(d["buys"], d["sells"])
    assert 0 <= out["pin"] <= 1
    assert 0 < out["alpha"] <= 0.9
    assert 0 <= out["delta"] <= 1


def test_validation() -> None:
    with pytest.raises(ValueError):
        pin_estimate(np.ones(5), np.ones(5))
    with pytest.raises(ValueError):
        pin_estimate(-np.ones(60), np.ones(60))
    with pytest.raises(ValueError):
        pin_estimate(np.ones(60) * np.nan, np.ones(60))


def test_deterministic() -> None:
    d = synth_pin(seed=5)
    a = pin_estimate(d["buys"], d["sells"])
    b = pin_estimate(d["buys"], d["sells"])
    assert a["pin"] == b["pin"]


def test_bench() -> None:
    out = bench_pin_model()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
