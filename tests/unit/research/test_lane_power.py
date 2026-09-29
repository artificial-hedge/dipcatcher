"""Pins for the sequential-power bench."""

from __future__ import annotations

import polars as pl

from quant_fund.research import lane_power
from quant_fund.research.lane_power import LANE_POWER_SCHEMA, lane_power_bench


def test_receipt_schema_and_digest() -> None:
    frame, receipt = lane_power_bench(defects=(0.0, 0.5), n_steps=64, n_seeds=3)
    assert receipt["schema"] == LANE_POWER_SCHEMA
    assert receipt["kind"] == "lane_power"
    assert isinstance(receipt["inputs_sha256"], str)
    assert set(receipt["params"]["lanes"]) == set(lane_power._LANES)


def test_missing_lanes_are_marked_not_fabricated() -> None:
    """On a checkout without the lane branches every lane reports
    lane_missing, n_lanes_ok is 0, and no alarm rows are fabricated."""
    frame, receipt = lane_power_bench(defects=(0.0,), n_steps=32, n_seeds=2)
    if receipt["n_lanes_ok"] == 0:
        assert set(frame["status"].unique()) == {"lane_missing"}
    else:
        ok = frame.filter(pl.col("status") == "ok")
        assert ok["alarmed"].dtype == pl.Boolean


def test_unknown_lane_reported() -> None:
    frame, _ = lane_power_bench(defects=(0.0,), n_steps=32, n_seeds=1, lanes=("no_such_lane",))
    assert frame["status"].unique().to_list() == ["unknown_lane"]


def test_runner_contract_when_lanes_present() -> None:
    """If a lane module is importable on this checkout, its runner must
    return a well-formed result at defect 0 without raising."""
    import importlib

    for lane, runner in lane_power._LANES.items():
        mod_name = {
            "coverage_watch": "quant_fund.research.coverage_watch",
            "tail_watch": "quant_fund.research.tail_watch",
            "calibration_eprocess": "quant_fund.research.calibration_eprocess",
            "drift_alarm": "quant_fund.research.drift_alarm",
            "loss_cs": "quant_fund.research.loss_cs",
            "changepoint_localize": "quant_fund.research.changepoint_localize",
        }[lane]
        try:
            importlib.import_module(mod_name)
        except ImportError:
            continue  # lane absent on this checkout — lane_missing path
        r = runner(0.0, 0, 100, 0.05)
        assert isinstance(r.alarmed, bool)
        assert isinstance(r.t_alarm, float)
        assert isinstance(r.stat, float)


def test_null_control_is_bounded() -> None:
    """For any available lane, the defect=0 false-alarm rate must sit
    near alpha, not near 1 — the bench's own honesty check."""
    frame, receipt = lane_power_bench(defects=(0.0,), n_steps=200, n_seeds=10, alpha=0.05)
    for lane, rate in receipt["null_alarm_rate"].items():
        assert rate <= 0.4, f"{lane} false-alarmed at {rate}"
