"""SYNTHETIC preparation/score/receipt checks, not empirical market evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from scripts.eval_blueprint_graph import (
    FEATURE_NAMES,
    EvaluationConfig,
    array_hash,
    build_causal_features,
    canonical_hash,
    paired_block_ci,
    prepare_panel,
    score_phase,
    select_assets,
    write_immutable_run,
)

from quant_fund.models.asset_graph import GraphForecast


def _bars(n: int = 100, assets: tuple[str, ...] = ("A", "B", "C")) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    records = []
    for rank, asset in enumerate(assets):
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        for i, price in enumerate(prices):
            clock = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i)
            records.append(
                {
                    "security_id": asset,
                    "event_time": clock,
                    "available_time": clock,
                    "close_total_return": float(price),
                    "source": "yahoo",
                    "revision_id": "YAHOO_VENDOR_ADJ",
                    "adv": float((len(assets) - rank) * 100),
                }
            )
    return pl.DataFrame(records)


def _config() -> EvaluationConfig:
    return replace(
        EvaluationConfig(),
        train_start="2020-01-01",
        train_end="2020-02-12",
        validation_start="2020-02-15",
        validation_end="2020-03-10",
        test_start="2020-03-13",
        test_end="2020-04-08",
        asset_count=2,
        min_selection_rows=30,
        min_phase_dates=5,
    )


def _write_inputs(root, bars: pl.DataFrame) -> None:
    (root / "silver").mkdir()
    (root / "gold").mkdir()
    bars.drop("adv").write_parquet(root / "silver/bars.parquet")
    bars.select("security_id", "event_time", "available_time", "adv").write_parquet(
        root / "silver/universe.parquet"
    )
    ordered = bars.sort(["security_id", "event_time"])
    ordered.select(
        "security_id",
        "event_time",
        pl.col("event_time").shift(-1).over("security_id").alias("label_end_time_1"),
        (pl.col("close_total_return").shift(-1).over("security_id") / pl.col("close_total_return"))
        .log()
        .alias("future_log_return_1"),
    ).write_parquet(root / "gold/labels.parquet")


def test_selection_cannot_follow_future_liquidity_or_unpublished_membership() -> None:
    bars = _bars()
    config = _config()
    before = select_assets(bars, config)
    changed = bars.with_columns(
        pl.when(pl.col("event_time") > datetime(2020, 2, 12, 23, tzinfo=UTC))
        .then(pl.when(pl.col("security_id") == "C").then(1e12).otherwise(1))
        .otherwise(pl.col("adv"))
        .alias("adv")
    )
    assert select_assets(changed, config) == before
    unavailable = bars.with_columns(
        pl.when(pl.col("security_id") == "A")
        .then(pl.col("event_time") + pl.duration(days=100))
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    assert [row["security_id"] for row in select_assets(unavailable, config)] == ["B", "C"]


def test_features_are_prefix_invariant_and_stamp_max_input_availability() -> None:
    bars = _bars(50, ("A",))
    original = build_causal_features(bars)
    poisoned = bars.with_columns(
        pl.when(pl.col("event_time") >= datetime(2020, 2, 1, tzinfo=UTC))
        .then(pl.col("close_total_return") * 1000)
        .otherwise(pl.col("close_total_return"))
        .alias("close_total_return")
    )
    altered = build_causal_features(poisoned)
    np.testing.assert_array_equal(
        original.select(FEATURE_NAMES).head(30).to_numpy(),
        altered.select(FEATURE_NAMES).head(30).to_numpy(),
    )
    delayed = bars.with_columns(
        pl.when(pl.col("event_time") == datetime(2020, 1, 25, tzinfo=UTC))
        .then(pl.col("available_time") + pl.duration(days=10))
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    features = build_causal_features(delayed)
    row = features.filter(pl.col("event_time") == datetime(2020, 1, 26, tzinfo=UTC)).row(
        0, named=True
    )
    assert row["_feature_available_time"] > row["event_time"]
    assert not any(
        name.startswith("future_") or name.startswith("label_end_") for name in FEATURE_NAMES
    )


def test_preparation_uses_endpoint_availability_and_identical_node_date_rows(tmp_path) -> None:
    bars = _bars()
    _write_inputs(tmp_path, bars)
    prepared = prepare_panel(tmp_path, _config())
    assert prepared.asset_ids == ("A", "B")
    for name, phase in prepared.phases.items():
        assert phase.features.shape[:2] == phase.targets.shape
        assert phase.features.shape[2] == 4
        assert np.isfinite(phase.features).all()
        assert np.all(
            phase.feature_available_times <= np.asarray(phase.timestamps, dtype=object)[:, None]
        )
        assert np.all(
            phase.target_available_times > np.asarray(phase.timestamps, dtype=object)[:, None]
        )
        assert all(
            (future - origin).days == 1
            for origin, future in zip(
                phase.timestamps, phase.target_available_times[:, 0], strict=True
            )
        )
        assert prepared.audit["phase_counts"][name]["balanced_node_rows"] == phase.targets.size


def test_corrupt_label_endpoint_or_value_fails_closed(tmp_path) -> None:
    _write_inputs(tmp_path, _bars())
    path = tmp_path / "gold/labels.parquet"
    labels = pl.read_parquet(path).with_columns(
        (pl.col("future_log_return_1") + 0.1).alias("future_log_return_1")
    )
    labels.write_parquet(path)
    with pytest.raises(ValueError, match="disagree with endpoint"):
        prepare_panel(tmp_path, _config())


def test_synthetic_sources_and_duplicate_identity_fail_closed(tmp_path) -> None:
    bars = _bars().with_columns(pl.lit("synthetic").alias("source"))
    _write_inputs(tmp_path, bars)
    with pytest.raises(ValueError, match="empirical protocol"):
        prepare_panel(tmp_path, _config())
    duplicate = pl.concat([_bars(30, ("A",)), _bars(30, ("A",)).head(1)])
    with pytest.raises(ValueError, match="duplicate"):
        build_causal_features(duplicate)


def test_insufficient_phase_or_selection_does_not_claim_completion(tmp_path) -> None:
    _write_inputs(tmp_path, _bars())
    with pytest.raises(ValueError, match="insufficient assets"):
        prepare_panel(tmp_path, replace(_config(), asset_count=4))
    with pytest.raises(ValueError, match="insufficient complete"):
        prepare_panel(tmp_path, replace(_config(), min_phase_dates=100))


def test_paired_block_interval_replays_deterministically_and_preserves_pairs() -> None:
    difference = np.linspace(-0.2, 0.1, 120)
    first = paired_block_ci(difference, block=20, samples=300, seed=7)
    assert first == paired_block_ci(difference, block=20, samples=300, seed=7)
    assert first["ci_low"] < first["mean_gcn_minus_baseline"] < first["ci_high"]
    constant = paired_block_ci(np.full(40, -0.2), block=20, samples=100, seed=7)
    assert constant["ci_low"] == pytest.approx(-0.2)
    assert constant["ci_high"] == pytest.approx(-0.2)
    with pytest.raises(ValueError):
        paired_block_ci(np.asarray([np.nan, 1]), block=20, samples=100, seed=7)


def test_scores_average_assets_then_dates_on_identical_rows() -> None:
    targets = np.asarray([[0.1, 0.2], [-0.1, 0.3], [0, -0.3]])
    clocks = tuple(datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(3))
    forecast = GraphForecast(np.zeros_like(targets), np.ones_like(targets), ("A", "B"), clocks)
    predictions = {
        name: forecast for name in ("gcn", "node_only", "pooled_gaussian", "ridge_gaussian")
    }
    report, losses = score_phase(
        predictions, targets, replace(EvaluationConfig(), bootstrap_samples=20)
    )
    assert report["models"]["gcn"]["crps"] == pytest.approx(
        forecast.crps(targets).mean(axis=1).mean()
    )
    for comparison in report["paired_date_block_ci"].values():
        assert comparison["crps"]["mean_gcn_minus_baseline"] == 0
    assert losses["gcn_crps"].shape == (3,)
    mismatched = dict(predictions)
    mismatched["node_only"] = GraphForecast(
        np.zeros_like(targets), np.ones_like(targets), ("B", "A"), clocks
    )
    with pytest.raises(ValueError, match="order must agree"):
        score_phase(mismatched, targets, EvaluationConfig())


def test_hashes_bind_shape_values_and_names() -> None:
    arrays = {"X": np.arange(12).reshape(3, 4)}
    digest = array_hash(arrays)
    assert digest == array_hash({"X": arrays["X"].astype(float)})
    assert digest != array_hash({"X": arrays["X"].reshape(4, 3)})
    assert digest != array_hash({"Y": arrays["X"]})
    assert digest != array_hash({"X": arrays["X"] + 1})


def test_immutable_receipt_binds_exact_side_artifact_bytes(tmp_path) -> None:
    output = tmp_path / "run"
    payload = {
        "synthetic": True,
        "research_only": True,
        "live_pnl_claim": False,
        "holdout_status": "SYNTHETIC_fixture",
    }
    receipt = write_immutable_run(output, payload, {"models.npz": b"fixture-parameters"})
    result = json.loads(receipt.read_text())
    assert result["artifacts"]["models.npz"] == hashlib.sha256(b"fixture-parameters").hexdigest()
    digest = result.pop("receipt_sha256")
    assert digest == canonical_hash(result)
    with pytest.raises(FileExistsError):
        write_immutable_run(output, payload, {"models.npz": b"different"})
