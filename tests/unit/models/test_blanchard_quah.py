"""Unit tests for quant_fund.models.blanchard_quah."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.blanchard_quah import (
    bench_blanchard_quah,
    blanchard_quah,
    synth_bq,
)


def test_demand_long_run_neutral() -> None:
    d = synth_bq(seed=1)
    out = blanchard_quah(d["y"])
    assert abs(out["long_run_demand_y0"]) < 1e-8


def test_supply_has_long_run_effect() -> None:
    d = synth_bq(seed=1)
    out = blanchard_quah(d["y"])
    assert abs(out["long_run_supply_y1"]) > 0.01


def test_schema() -> None:
    d = synth_bq(seed=2)
    out = blanchard_quah(d["y"])
    assert set(out) == {
        "a1_00",
        "long_run_supply_y0",
        "long_run_demand_y0",
        "long_run_supply_y1",
        "cum_demand_response_h",
        "cum_supply_response_h",
        "shock_corr",
    }


def test_determinism() -> None:
    d = synth_bq(seed=5)
    a = blanchard_quah(d["y"])
    b = blanchard_quah(d["y"])
    assert a == b


def test_validation() -> None:
    with pytest.raises(ValueError):
        blanchard_quah(np.ones((30, 2)))  # too short
    with pytest.raises(ValueError):
        blanchard_quah(np.ones((200, 3)))  # wrong width
    with pytest.raises(ValueError):
        blanchard_quah(np.full((200, 2), np.nan))
    with pytest.raises(ValueError):
        blanchard_quah(synth_bq(seed=0)["y"], horizon=1)


def test_bench_keys_and_pass() -> None:
    out = bench_blanchard_quah()
    assert set(out) == {
        "synthetic_detects",
        "synthetic_determinism",
        "synthetic_lr_demand",
        "synthetic_lr_supply_y1",
        "synthetic_cum_demand",
        "synthetic_cum_supply",
        "synthetic_a1_00",
    }
    assert all(np.isfinite(v) for v in out.values())
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
