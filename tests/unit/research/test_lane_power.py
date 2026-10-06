"""Pins for the sequential-power bench."""

from __future__ import annotations

import polars as pl
import pytest

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
        mod_name = lane_power._LANE_MODULES[lane]
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


def test_dataset_sha256_tracks_streams_not_alpha() -> None:
    """dataset_sha256 digests the exact per-cell update streams: identical
    (lane, defect, seed) grids agree regardless of alpha; a different
    defect grid digests different streams."""
    _, r1 = lane_power_bench(
        defects=(0.0, 0.5), n_steps=48, n_seeds=3, alpha=0.05, lanes=("drift_alarm",)
    )
    if r1["n_lanes_ok"] == 0:
        pytest.skip("drift_alarm lane absent on this checkout")
    _, r2 = lane_power_bench(
        defects=(0.0, 0.5), n_steps=48, n_seeds=3, alpha=0.1, lanes=("drift_alarm",)
    )
    _, r3 = lane_power_bench(
        defects=(0.0, 0.75), n_steps=48, n_seeds=3, alpha=0.05, lanes=("drift_alarm",)
    )
    d1, d2, d3 = (r["dataset_sha256"] for r in (r1, r2, r3))
    assert len(d1) == 64 and all(c in "0123456789abcdef" for c in d1)
    assert d1 == d2  # alpha is a run param, not data
    assert d1 != d3


# Monitor-family modules that are deliberately NOT power-bench lanes —
# each exclusion is annotated; merging a lane-shaped module without
# registering it in _LANES fails this suite.
_EXCLUDED_LANE_MODULES = {
    "emerge": "e-value mergers — primitives, not a monitor lane",
    "lane_power": "the bench itself",
    "monitor_run": "the runner that drives lanes, not a lane",
    "verdict_run": "stream producer for honest_verdict",
    "honest_verdict": "composite claim over lanes — measured via components",
    "evalue_contracts": "verifier contracts, not a monitor lane",
    "policy_eprocess": "paired-episode dominance lane — measured by its own "
    "policy_eprocess_bench, not the single-stream power bench",
    "corpus_inference": "operates on the receipt corpus, not a stream",
    "online_fdr": "operates on the receipt corpus, not a stream",
    "winner_curse": "batch correction, not sequential",
    "fleet_race": "head-elimination driver — uses LossEProcess internally",
}

_LANE_SUFFIXES = ("_watch", "_eprocess", "_alarm", "_monitor", "_localize", "_cs")
_LANE_NAMES = ("evalues", "emerge")


def test_monitor_lane_completeness_ratchet() -> None:
    """Every monitor-family module in research/ must be a registered
    power-bench lane or carry an explicit exclusion — a new sequential
    lane cannot land silently unmeasured by lane_power."""
    import pathlib

    research_dir = pathlib.Path(lane_power.__file__).resolve().parent
    discovered: set[str] = set()
    for path in research_dir.glob("*.py"):
        stem = path.stem
        if stem in _LANE_NAMES or stem.endswith(_LANE_SUFFIXES):
            discovered.add(stem)
    registered = {mod.rsplit(".", 1)[-1] for mod in lane_power._LANE_MODULES.values()}
    unregistered = discovered - registered - set(_EXCLUDED_LANE_MODULES)
    assert not unregistered, (
        f"monitor-family module(s) not in lane_power._LANES and not "
        f"excluded: {sorted(unregistered)} — add a runner or a "
        f"_EXCLUDED_LANE_MODULES entry"
    )


def test_lane_power_receipt_v2_round_trip(tmp_path) -> None:
    """receipt_version=2 seals the lane_power.v1 body in the envelope."""
    import json
    from pathlib import Path

    from quant_fund.research.lane_power import write_lane_power_receipt
    from quant_fund.research.receipt_v2 import verify_receipt_file

    assert isinstance(tmp_path, Path)
    _, receipt = lane_power_bench(defects=(0.0,), n_steps=32, n_seeds=2)
    path = write_lane_power_receipt(receipt, tmp_path, receipt_version=2)
    payload = json.loads(path.read_text())
    assert payload["schema"] == "receipt.v2"
    assert payload["payload"]["kind"] == "lane_power"
    assert payload["payload"]["inputs_sha256"] == receipt["inputs_sha256"]
    assert verify_receipt_file(path)["valid"] is True
