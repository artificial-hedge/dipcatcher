import numpy as np
import pytest

from quant_fund.models.roy_model import bench_roy_model, roy_fit, synth_roy


def test_roy_recovers_sector1_price() -> None:
    d = synth_roy(price1=0.5, price0=0.3, seed=2)
    out = roy_fit(d["wage"], d["sector"], d["x"])
    assert abs(out["skill_price1"] - 0.5) < 0.2


def test_roy_positive_selection() -> None:
    d = synth_roy(seed=3)
    out = roy_fit(d["wage"], d["sector"], d["x"])
    assert out["rho1_mills"] > 0.2


def test_roy_naive_gap_overstates() -> None:
    d = synth_roy(seed=4)
    out = roy_fit(d["wage"], d["sector"], d["x"])
    assert out["gap_naive"] > out["gap_corrected"] + 0.3


def test_roy_validation() -> None:
    d = synth_roy(seed=5)
    with pytest.raises(ValueError):
        roy_fit(d["wage"][:100], d["sector"][:100], d["x"][:100])
    with pytest.raises(ValueError):
        roy_fit(d["wage"], d["sector"] + 2.0, d["x"])
    with pytest.raises(ValueError):
        roy_fit(d["wage"][:-1], d["sector"], d["x"])
    with pytest.raises(ValueError):
        roy_fit(np.where(d["sector"] == 1.0, np.nan, d["wage"]), d["sector"], d["x"])


def test_roy_deterministic() -> None:
    d = synth_roy(seed=6)
    a = roy_fit(d["wage"], d["sector"], d["x"])
    b = roy_fit(d["wage"], d["sector"], d["x"])
    assert a["skill_price1"] == b["skill_price1"]


def test_bench_roy_model() -> None:
    out = bench_roy_model()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
