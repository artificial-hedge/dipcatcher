"""Adversarial audit lane for the fx1 forecast data layer.

Probes :mod:`fx1.forecast.schema`, :mod:`fx1.forecast.features`, and
:mod:`fx1.forecast.artifacts` against their stated contracts: fail-closed
schema validation, point-in-time causality (no look-ahead), artifact digest
binding, determinism, and type/bounds enforcement. Surfaces that are
tolerated by design are pinned as named ``flag`` probes rather than silently
accepted; a ``flag`` verdict is a documented observation, not a failure.

Defects this lane fixed in-module — each pinned by a ``fixed`` probe that
flips to ``fail`` if the behavior regresses:

- ``visible_bars`` (features.py): ``decision_time=None`` early-returned
  *before* the lineage checks, so null ``available_time``/``event_time`` and
  ``event_time > available_time`` rows passed through; the full-PIT-column
  path also dropped null-availability rows silently while the sparse path
  raised. Lineage is now checked on every path, before the cutoff.
- ``resample_ohlcv`` (features.py): a bar released before its own event could
  hide inside a bucket whose aggregate max ``available_time`` still passed,
  and a bucket mixing ``revision_id`` values silently collapsed to the first
  while a ``source`` mix raised. Both are refused up front now. Duplicate
  input bar keys are also refused before aggregation can hide them.
- ``validate_forecast_schema`` (schema.py): quantile ordering was only
  checked on rows where *every* quantile was non-null — a crossing could
  hide behind one missing value. Present adjacent pairs are now ordered on
  every row.

Receipt: ``forecast_data_audit()`` returns the unsealed body;
``forecast_data_audit_bench()`` seals it with ``receipt_sha256`` over the
canonical JSON bytes. Synthetic fixtures only; no market evidence, no live
claims.
"""

from __future__ import annotations

import json
import pickle
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import polars as pl

from fx1.forecast.artifacts import (
    UntrustedArtifactError,
    import_optional,
    load_artifact,
    probe_artifact,
)
from fx1.forecast.features import (
    OhlcvFeaturePipeline,
    resample_ohlcv,
    visible_bars,
)
from fx1.forecast.schema import (
    SchemaError,
    validate_feature_schema,
    validate_forecast_schema,
)
from quant_fund.schemas.errors import LeakageError, PointInTimeError
from quant_fund.utils.atomicio import atomic_write_text
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes, hash_file
from quant_fund.utils.reproducibility import git_revision

AUDIT_KIND = "forecast_data_audit"
AUDIT_SCHEMA = "forecast_data_audit.v1"
AUDIT_RECEIPT_NAME = "forecast_data_audit.json"

Verdict = Literal["pass", "fail", "fixed", "flag"]

_PASSING_VERDICTS = ("pass", "fixed", "flag")


def _stamp(i: int) -> datetime:
    return datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i)


def _panel(
    n: int = 30,
    names: tuple[str, ...] = ("AAA", "BBB"),
    *,
    pit: bool = True,
) -> pl.DataFrame:
    """Deterministic OHLCV panel with valid lineage stamps."""
    rows: list[dict[str, object]] = []
    for s_i, sid in enumerate(names):
        price = 40.0 + 5.0 * s_i
        for i in range(n):
            price *= 1.01 if (i + s_i) % 2 == 0 else 0.99
            row: dict[str, object] = {
                "security_id": sid,
                "event_time": _stamp(i),
                "available_time": _stamp(i),
                "open": price * 0.99,
                "high": price * 1.02,
                "low": price * 0.98,
                "close": price,
                "volume": 1_000.0 + i,
            }
            if pit:
                row["ingested_time"] = _stamp(i)
                row["source"] = "synthetic"
                row["revision_id"] = "v1"
            rows.append(row)
    return pl.DataFrame(rows)


def _features() -> tuple[OhlcvFeaturePipeline, pl.DataFrame]:
    pipeline = OhlcvFeaturePipeline([1, 5], vol_window=5)
    return pipeline, pipeline.build(_panel(), decision_time=_stamp(29))


def _forecast() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [_stamp(0)],
            "security_id": ["AAA"],
            "horizon_bars": pl.Series([1], dtype=pl.Int64),
            "predicted_return": [0.01],
            "predicted_price": [41.0],
            "confidence": [0.5],
            "q_0.1": [0.0],
            "q_0.5": [0.01],
            "q_0.9": [0.2],
        }
    )


def _raises(fn: Callable[[], object], exc: type[Exception]) -> tuple[bool, str]:
    """Exact-type error check: a subclass or different type is a wrong error."""
    try:
        fn()
    except Exception as raised:
        if type(raised) is exc:
            return True, f"{type(raised).__name__}: {raised}"
        return False, f"wrong error {type(raised).__name__}: {raised}"
    return False, "no error raised"


def _contract(probe: str, area: str, ok: bool, detail: str) -> dict[str, str]:
    return {
        "probe": probe,
        "area": area,
        "verdict": "pass" if ok else "fail",
        "detail": detail,
    }


def _expect_error(
    probe: str,
    area: str,
    fn: Callable[[], object],
    exc: type[Exception],
    *,
    verdict: Verdict = "pass",
    scrub: tuple[str, ...] = (),
) -> dict[str, str]:
    ok, detail = _raises(fn, exc)
    for token in scrub:
        detail = detail.replace(token, "<scrubbed>")
    return {
        "probe": probe,
        "area": area,
        "verdict": verdict if ok else "fail",
        "detail": detail,
    }


def _flag(probe: str, area: str, detail: str) -> dict[str, str]:
    return {"probe": probe, "area": area, "verdict": "flag", "detail": detail}


# ---------------------------------------------------------------------------
# schema.py — feature-frame contract
# ---------------------------------------------------------------------------


def _schema_feature_probes() -> list[dict[str, str]]:
    pipeline, clean = _features()
    columns = pipeline.feature_columns()
    area = "schema.feature"
    return [
        _expect_error(
            "empty_frame_rejected",
            area,
            lambda: validate_feature_schema(clean.head(0), columns),
            SchemaError,
        ),
        _expect_error(
            "missing_required_key_rejected",
            area,
            lambda: validate_feature_schema(clean.drop("event_time"), columns),
            SchemaError,
        ),
        _expect_error(
            "lookahead_column_raises_leakage",
            area,
            lambda: validate_feature_schema(
                clean.with_columns(pl.lit(0.0).alias("future_close")), columns
            ),
            LeakageError,
        ),
        _expect_error(
            "label_like_prefix_raises_leakage",
            area,
            lambda: validate_feature_schema(
                clean.with_columns(pl.lit(0.0).alias("fwd_ret")), columns
            ),
            LeakageError,
        ),
        _expect_error(
            "undeclared_extra_column_rejected",
            area,
            lambda: validate_feature_schema(
                clean.with_columns(pl.lit(0.0).alias("aux_score")), columns
            ),
            SchemaError,
        ),
        _expect_error(
            "duplicate_key_rows_rejected",
            area,
            lambda: validate_feature_schema(pl.concat([clean, clean.head(1)]), columns),
            SchemaError,
        ),
        _expect_error(
            "non_numeric_feature_dtype_rejected",
            area,
            lambda: validate_feature_schema(
                clean.with_columns(pl.lit("x").alias("ret_1")), columns
            ),
            SchemaError,
        ),
        _expect_error(
            "nonfinite_feature_rejected",
            area,
            lambda: validate_feature_schema(
                clean.with_columns(pl.lit(float("inf")).alias("ret_1")), columns
            ),
            SchemaError,
        ),
        _expect_error(
            "null_feature_rejected",
            area,
            lambda: validate_feature_schema(
                clean.with_columns(
                    pl.when(pl.col("event_time") == clean["event_time"].max())
                    .then(None)
                    .otherwise(pl.col("ret_1"))
                    .alias("ret_1")
                ),
                columns,
            ),
            SchemaError,
        ),
        _contract(
            "clean_frame_accepted",
            area,
            not _raises(lambda: validate_feature_schema(clean, columns), Exception)[0],
            "declared numeric features pass the allow-list",
        ),
        _flag(
            "narrow_numeric_dtype_allowlist",
            area,
            "feature dtypes limited to Float32/Float64/Int32/Int64 — UInt8/16 "
            "features are rejected; over-strict in the safe direction",
        ),
        _flag(
            "context_columns_dtype_unchecked",
            area,
            "close/security_id/event_time/available_time dtypes are not checked "
            "by the schema — a Utf8 close passes validation (it crashes upstream "
            "in the pipeline, never at predict time)",
        ),
    ]


# ---------------------------------------------------------------------------
# schema.py — forecast-frame contract
# ---------------------------------------------------------------------------


def _schema_forecast_probes() -> list[dict[str, str]]:
    clean = _forecast()
    area = "schema.forecast"
    no_pred = clean.drop(["predicted_return", "predicted_price", "q_0.1", "q_0.5", "q_0.9"])
    partial_null_crossing = pl.DataFrame(
        {
            "event_time": [_stamp(0), _stamp(1)],
            "security_id": ["AAA", "AAA"],
            "horizon_bars": pl.Series([1, 1], dtype=pl.Int64),
            "predicted_return": [0.01, 0.02],
            "q_0.1": [None, 0.0],
            "q_0.5": [0.05, 0.5],
            "q_0.9": [0.04, 0.9],
        }
    )
    return [
        _expect_error(
            "empty_frame_rejected",
            area,
            lambda: validate_forecast_schema(clean.head(0)),
            SchemaError,
        ),
        _expect_error(
            "missing_key_rejected",
            area,
            lambda: validate_forecast_schema(clean.drop("horizon_bars")),
            SchemaError,
        ),
        _expect_error(
            "no_prediction_column_rejected",
            area,
            lambda: validate_forecast_schema(no_pred),
            SchemaError,
        ),
        _expect_error(
            "label_column_raises_leakage",
            area,
            lambda: validate_forecast_schema(
                clean.with_columns(pl.lit(0.0).alias("realized_return"))
            ),
            LeakageError,
        ),
        _expect_error(
            "duplicate_keys_rejected",
            area,
            lambda: validate_forecast_schema(pl.concat([clean, clean])),
            SchemaError,
        ),
        _expect_error(
            "horizon_zero_rejected",
            area,
            lambda: validate_forecast_schema(
                clean.with_columns(pl.lit(0).cast(pl.Int64).alias("horizon_bars"))
            ),
            SchemaError,
        ),
        _expect_error(
            "horizon_null_rejected",
            area,
            lambda: validate_forecast_schema(
                clean.with_columns(pl.lit(None).cast(pl.Int64).alias("horizon_bars"))
            ),
            SchemaError,
        ),
        _expect_error(
            "horizon_string_dtype_rejected",
            area,
            lambda: validate_forecast_schema(clean.with_columns(pl.lit("1").alias("horizon_bars"))),
            SchemaError,
        ),
        _expect_error(
            "both_predictions_null_rejected",
            area,
            lambda: validate_forecast_schema(
                clean.with_columns(
                    pl.lit(None).cast(pl.Float64).alias("predicted_return"),
                    pl.lit(None).cast(pl.Float64).alias("predicted_price"),
                )
            ),
            SchemaError,
        ),
        _expect_error(
            "nonfinite_predicted_return_rejected",
            area,
            lambda: validate_forecast_schema(
                clean.with_columns(pl.lit(float("nan")).alias("predicted_return"))
            ),
            SchemaError,
        ),
        _expect_error(
            "nonpositive_predicted_price_rejected",
            area,
            lambda: validate_forecast_schema(
                clean.with_columns(
                    pl.lit(None).cast(pl.Float64).alias("predicted_return"),
                    pl.lit(-1.0).alias("predicted_price"),
                )
            ),
            SchemaError,
        ),
        _expect_error(
            "confidence_out_of_band_rejected",
            area,
            lambda: validate_forecast_schema(clean.with_columns(pl.lit(1.5).alias("confidence"))),
            SchemaError,
        ),
        _expect_error(
            "confidence_nan_rejected",
            area,
            lambda: validate_forecast_schema(
                clean.with_columns(pl.lit(float("nan")).alias("confidence"))
            ),
            SchemaError,
        ),
        _expect_error(
            "quantile_crossing_rejected",
            area,
            lambda: validate_forecast_schema(
                clean.with_columns(pl.lit(0.4).alias("q_0.1"), pl.lit(0.1).alias("q_0.9"))
            ),
            SchemaError,
        ),
        _expect_error(
            "quantile_ordering_partial_null_row",
            area,
            lambda: validate_forecast_schema(partial_null_crossing),
            SchemaError,
            verdict="fixed",
        ),
        _expect_error(
            "quantile_tau_out_of_band_rejected",
            area,
            lambda: validate_forecast_schema(clean.with_columns(pl.lit(0.0).alias("q_0.0"))),
            SchemaError,
        ),
        _contract(
            "clean_ladder_accepted",
            area,
            not _raises(lambda: validate_forecast_schema(clean), Exception)[0],
            "ordered quantiles, bounded confidence, both predictions present",
        ),
        _flag(
            "horizon_dtype_allowlist_partial",
            area,
            "horizon_bars accepts Int32/64 + UInt32/64 but not Int8/16 or "
            "UInt8/16 despite the 'non-null integer' message — over-strict in "
            "the safe direction (pinned in docs/AUDIT_FX1.md round 3)",
        ),
        _flag(
            "predicted_return_below_minus_one_accepted",
            area,
            "a predicted_return of -3.5 (a negative future price) passes — "
            "the positivity bound exists only on predicted_price",
        ),
        _flag(
            "extra_columns_tolerated",
            area,
            "forecast frames have no allow-list; an unknown column like "
            "aux_score rides into evaluation (label-like names still raise)",
        ),
        _flag(
            "noncanonical_quantile_name_tolerated",
            area,
            "q_50 or q_5 do not match ^q_(0.X+)$ — treated as plain extra "
            "columns, escaping the ordering check",
        ),
    ]


# ---------------------------------------------------------------------------
# features.py — point-in-time visibility and causality
# ---------------------------------------------------------------------------


def _visibility_probes() -> list[dict[str, str]]:
    area = "features.pit"
    bars = _panel(pit=False)
    cutoff = _stamp(15)

    corrupt_release = bars.with_columns(
        pl.when(pl.col("event_time") == _stamp(2))
        .then(pl.lit(_stamp(1)))
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    null_avail = bars.with_columns(
        pl.when(pl.col("event_time") == _stamp(2))
        .then(None)
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    null_avail_pit = _panel().with_columns(
        pl.when(pl.col("event_time") == _stamp(2))
        .then(None)
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    null_event = bars.with_columns(
        pl.when(pl.col("event_time") == _stamp(2))
        .then(None)
        .otherwise(pl.col("event_time"))
        .alias("event_time")
    )

    delayed = bars.with_columns(
        pl.when(pl.col("event_time") == _stamp(2))
        .then(pl.lit(_stamp(6)))
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    removed = bars.filter(pl.col("event_time") != _stamp(2))
    hidden = visible_bars(delayed, _stamp(5))
    withheld = visible_bars(removed, _stamp(5))

    return [
        _expect_error(
            "missing_available_time_rejected",
            area,
            lambda: visible_bars(bars.drop("available_time"), cutoff),
            PointInTimeError,
        ),
        _expect_error(
            "missing_event_time_rejected",
            area,
            lambda: visible_bars(bars.drop("event_time"), cutoff),
            PointInTimeError,
            verdict="fixed",
        ),
        _expect_error(
            "null_availability_rejected_no_cutoff",
            area,
            lambda: visible_bars(null_avail, None),
            PointInTimeError,
            verdict="fixed",
        ),
        _expect_error(
            "null_availability_rejected_pit_path",
            area,
            lambda: visible_bars(null_avail_pit, cutoff),
            PointInTimeError,
            verdict="fixed",
        ),
        _expect_error(
            "release_before_event_rejected_no_cutoff",
            area,
            lambda: visible_bars(corrupt_release, None),
            PointInTimeError,
            verdict="fixed",
        ),
        _expect_error(
            "release_before_event_rejected_with_cutoff",
            area,
            lambda: visible_bars(corrupt_release, cutoff),
            PointInTimeError,
        ),
        _expect_error(
            "null_event_time_rejected",
            area,
            lambda: visible_bars(null_event, cutoff),
            PointInTimeError,
            verdict="fixed",
        ),
        _contract(
            "event_after_cutoff_excluded",
            area,
            visible_bars(bars, cutoff)["event_time"].max() == cutoff
            and visible_bars(bars, cutoff).height < bars.height,
            "rows with event_time > decision_time are dropped",
        ),
        _contract(
            "late_release_equals_absent_bar",
            area,
            bool(hidden.equals(withheld)),
            "a bar whose release is after the decision is identical to absence",
        ),
        _contract(
            "boundary_release_visible",
            area,
            visible_bars(bars.filter(pl.col("event_time") == cutoff), cutoff).height == 2,
            "event_time == available_time == decision_time is observable",
        ),
        _flag(
            "no_cutoff_returns_all_bars",
            area,
            "decision_time=None applies lineage checks but no visibility "
            "filter — callers get every valid row (fit-time semantics)",
        ),
        _flag(
            "naive_timestamp_assumed_utc",
            area,
            "as_utc stamps a naive datetime as UTC instead of refusing; a "
            "local-time stamp upstream would be silently reinterpreted",
        ),
        _flag(
            "secondary_lineage_nulls_tolerated",
            area,
            "only event_time/available_time are enforced non-null; "
            "ingested_time/source/revision_id may carry nulls",
        ),
    ]


def _causality_probes() -> list[dict[str, str]]:
    area = "features.causality"
    pipeline = OhlcvFeaturePipeline([1, 5], vol_window=5)
    bars = _panel()
    decision = _stamp(20)
    original = pipeline.build(bars, decision_time=decision)

    future = bars.with_columns(
        pl.when(pl.col("event_time") > decision)
        .then(pl.col("close") * 10.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    rebuilt = pipeline.build(future, decision_time=decision)

    at_decision = (pl.col("event_time") == pl.lit(decision)) & (
        pl.col("security_id") == pl.lit("AAA")
    )
    row = original.filter(at_decision).row(0, named=True)
    closes = bars.filter(pl.col("security_id") == "AAA").sort("event_time")
    close_t = closes.filter(pl.col("event_time") == _stamp(20))["close"][0]
    close_prev = closes.filter(pl.col("event_time") == _stamp(19))["close"][0]
    close_m5 = closes.filter(pl.col("event_time") == _stamp(15))["close"][0]

    # Perturbing a close *before* the trailing windows must not move any
    # feature at the decision row (ret_1@20, mom_5@20, vol_5@20 all end at 20).
    pre_window = bars.with_columns(
        pl.when((pl.col("event_time") == _stamp(10)) & (pl.col("security_id") == "AAA"))
        .then(pl.col("close") * 50.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    perturbed_row = (
        pipeline.build(pre_window, decision_time=decision).filter(at_decision).row(0, named=True)
    )
    trailing_only = all(perturbed_row[name] == row[name] for name in pipeline.feature_columns())

    shuffled = bars.sample(fraction=1.0, shuffle=True, seed=7)
    rebuilt_shuffle = pipeline.build(shuffled, decision_time=decision)

    declared = set(pipeline.feature_columns()) | {
        "event_time",
        "security_id",
        "close",
        "available_time",
        "decision_time",
    }

    return [
        _contract(
            "future_close_invisible",
            area,
            bool(rebuilt.equals(original)),
            "perturbing closes after the decision time changes no feature row",
        ),
        _contract(
            "ret_1_exact_trailing_return",
            area,
            abs(row["ret_1"] - (close_t / close_prev - 1.0)) < 1e-12,
            f"ret_1={row['ret_1']:.12f} vs manual {close_t / close_prev - 1.0:.12f}",
        ),
        _contract(
            "mom_5_exact_trailing_return",
            area,
            abs(row["mom_5"] - (close_t / close_m5 - 1.0)) < 1e-12,
            f"mom_5={row['mom_5']:.12f} vs manual {close_t / close_m5 - 1.0:.12f}",
        ),
        _contract(
            "features_use_only_trailing_window",
            area,
            trailing_only,
            "perturbing a close before the rolling window leaves the "
            "decision-row features bit-identical",
        ),
        _contract(
            "row_order_invariant",
            area,
            bool(rebuilt_shuffle.equals(original)),
            "shuffled input produces identical feature rows",
        ),
        _contract(
            "only_declared_columns_emitted",
            area,
            set(original.columns) <= declared,
            f"emitted columns {sorted(original.columns)}",
        ),
        _contract(
            "all_features_finite",
            area,
            all(bool(original[name].is_finite().all()) for name in pipeline.feature_columns()),
            "no inf/NaN in any emitted feature column",
        ),
        _expect_error(
            "duplicate_bars_rejected",
            area,
            lambda: pipeline.build(pl.concat([bars, bars.head(1)]), decision_time=_stamp(29)),
            PointInTimeError,
        ),
        _expect_error(
            "empty_lookbacks_rejected",
            area,
            lambda: OhlcvFeaturePipeline([], vol_window=5),
            ValueError,
        ),
        _expect_error(
            "duplicate_lookbacks_rejected",
            area,
            lambda: OhlcvFeaturePipeline([5, 5], vol_window=5),
            ValueError,
        ),
        _expect_error(
            "vol_window_below_two_rejected",
            area,
            lambda: OhlcvFeaturePipeline([1], vol_window=1),
            ValueError,
        ),
        _flag(
            "empty_feature_frame_silent",
            area,
            "a panel shorter than the longest lookback returns an empty frame "
            "without SchemaError — the runner tolerates empty slices",
        ),
    ]


def _resample_probes() -> list[dict[str, str]]:
    area = "features.resample"
    day = datetime(2020, 1, 2, tzinfo=UTC)

    def bar(
        hour: int,
        close: float,
        *,
        avail: datetime | None = None,
        source: str = "fixture",
        revision: str = "v1",
    ) -> dict[str, object]:
        stamp = day.replace(hour=hour)
        return {
            "security_id": "AAA",
            "event_time": stamp,
            "available_time": avail if avail is not None else stamp,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 5.0,
            "source": source,
            "revision_id": revision,
        }

    intra = pl.DataFrame([bar(9, 10.0), bar(10, 11.0), bar(14, 12.0)])
    out = resample_ohlcv(intra, "1d")
    out_row = out.row(0, named=True)

    mixed_source = pl.DataFrame([bar(9, 10.0), bar(10, 11.0, source="other")])
    mixed_revision = pl.DataFrame([bar(9, 10.0), bar(10, 11.0, revision="v2")])
    hidden_corrupt = pl.DataFrame(
        [
            bar(9, 10.0, avail=day.replace(hour=15)),
            bar(14, 12.0, avail=day.replace(hour=13)),
        ]
    )
    duplicate_key = pl.DataFrame([bar(9, 10.0), bar(9, 11.0)])
    shuffled = intra.sample(fraction=1.0, shuffle=True, seed=3)
    late_release = resample_ohlcv(
        pl.DataFrame([bar(9, 10.0, avail=day.replace(hour=10)), bar(14, 12.0)]),
        "1d",
    ).row(0, named=True)

    return [
        _contract(
            "bucket_labeled_by_last_print",
            area,
            out_row["event_time"] == day.replace(hour=14),
            "bucket timestamp is the last source event_time, not bucket open",
        ),
        _contract(
            "ohlcv_aggregation_correct",
            area,
            out_row["open"] == 9.5
            and out_row["high"] == 13.0
            and out_row["low"] == 9.0
            and out_row["close"] == 12.0
            and out_row["volume"] == 15.0,
            f"open={out_row['open']} high={out_row['high']} low={out_row['low']} "
            f"close={out_row['close']} volume={out_row['volume']}",
        ),
        _contract(
            "bucket_availability_is_max",
            area,
            late_release["available_time"] == day.replace(hour=14),
            "bucket is observable only at its last print's release",
        ),
        _contract(
            "row_order_invariant",
            area,
            bool(resample_ohlcv(shuffled, "1d").equals(out)),
            "shuffled source rows produce identical buckets",
        ),
        _expect_error(
            "missing_column_rejected",
            area,
            lambda: resample_ohlcv(intra.drop("close"), "1d"),
            PointInTimeError,
        ),
        _expect_error(
            "empty_every_rejected",
            area,
            lambda: resample_ohlcv(intra, ""),
            ValueError,
        ),
        _expect_error(
            "mixed_sources_refused",
            area,
            lambda: resample_ohlcv(mixed_source, "1d"),
            PointInTimeError,
        ),
        _expect_error(
            "mixed_revisions_refused",
            area,
            lambda: resample_ohlcv(mixed_revision, "1d"),
            PointInTimeError,
            verdict="fixed",
        ),
        _expect_error(
            "per_bar_release_before_event_refused",
            area,
            lambda: resample_ohlcv(hidden_corrupt, "1d"),
            PointInTimeError,
            verdict="fixed",
        ),
        _expect_error(
            "null_lineage_refused",
            area,
            lambda: resample_ohlcv(
                intra.with_columns(
                    pl.when(pl.col("event_time") == day.replace(hour=10))
                    .then(None)
                    .otherwise(pl.col("available_time"))
                    .alias("available_time")
                ),
                "1d",
            ),
            PointInTimeError,
            verdict="fixed",
        ),
        _expect_error(
            "duplicate_keys_refused",
            area,
            lambda: resample_ohlcv(duplicate_key, "1d"),
            PointInTimeError,
            verdict="fixed",
        ),
        _flag(
            "invalid_every_raises_polars_error",
            area,
            "every='bogus' raises a raw polars InvalidOperationError, not a "
            "typed PointInTimeError — loud crash, still fail-closed",
        ),
    ]


# ---------------------------------------------------------------------------
# artifacts.py — checkpoint trust and digest binding
# ---------------------------------------------------------------------------


def _artifact_probes() -> list[dict[str, str]]:
    area = "artifacts"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        json_path = root / "model.json"
        payload = {"horizon_bars": 3, "version": "audit-1"}
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        digest = hash_file(json_path)

        loaded = load_artifact(json_path)
        json_ok = (
            loaded.format == "json"
            and loaded.sha256 == digest
            and loaded.payload == payload
            and loaded.version == "audit-1"
        )

        pkl_path = root / "model.pkl"
        pkl_path.write_bytes(pickle.dumps({"horizon_bars": 5}))
        pkl_digest = hash_file(pkl_path)

        loaded_pkl = load_artifact(
            pkl_path,
            allow_unsafe_deserialization=True,
            trusted_checkpoint_sha256=pkl_digest,
        )
        pkl_ok = loaded_pkl.payload == {"horizon_bars": 5} and loaded_pkl.sha256 == pkl_digest

        probe = probe_artifact(pkl_path)
        probe_ok = (
            probe["sha256"] == pkl_digest
            and probe["format"] == "pickle"
            and probe["version"] is None
        )

        unknown_path = root / "model.bin"
        unknown_path.write_bytes(b"00")

        torch_ok = True
        torch_detail = "torch weights_only load accepted"
        torch_path = root / "model.pt"
        try:
            torch = import_optional("torch")
            torch.save({"w": torch.tensor([1.0, 2.0])}, torch_path)
            loaded_torch = load_artifact(torch_path)
            torch_ok = loaded_torch.format == "torch" and loaded_torch.sha256 == hash_file(
                torch_path
            )
            torch_detail = "weights_only=True load stamped with the file digest"
        except Exception as exc:  # backend genuinely unavailable
            torch_detail = f"torch unavailable, path not exercised: {type(exc).__name__}"

        return [
            _expect_error(
                "missing_checkpoint_fails_closed",
                area,
                lambda: load_artifact(root / "nope.pkl"),
                FileNotFoundError,
                scrub=(str(root),),
            ),
            _expect_error(
                "unknown_suffix_rejected",
                area,
                lambda: load_artifact(unknown_path),
                ValueError,
            ),
            _contract(
                "json_roundtrip_stamps_digest",
                area,
                json_ok,
                "payload, format, sidecar version, and sha256 all bound",
            ),
            _expect_error(
                "safe_format_wrong_digest_refused",
                area,
                lambda: load_artifact(json_path, trusted_checkpoint_sha256="0" * 64),
                UntrustedArtifactError,
            ),
            _contract(
                "safe_format_correct_digest_accepted",
                area,
                not _raises(
                    lambda: load_artifact(json_path, trusted_checkpoint_sha256=digest),
                    Exception,
                )[0],
                "digest binding enforced even without unsafe opt-in",
            ),
            _expect_error(
                "unsafe_pickle_requires_opt_in",
                area,
                lambda: load_artifact(pkl_path),
                UntrustedArtifactError,
            ),
            _expect_error(
                "unsafe_opt_in_still_requires_digest",
                area,
                lambda: load_artifact(pkl_path, allow_unsafe_deserialization=True),
                UntrustedArtifactError,
            ),
            _expect_error(
                "digest_format_enforced",
                area,
                lambda: load_artifact(
                    pkl_path,
                    allow_unsafe_deserialization=True,
                    trusted_checkpoint_sha256="ZZZ",
                ),
                UntrustedArtifactError,
            ),
            _expect_error(
                "digest_mismatch_refused",
                area,
                lambda: load_artifact(
                    pkl_path,
                    allow_unsafe_deserialization=True,
                    trusted_checkpoint_sha256="1" * 64,
                ),
                UntrustedArtifactError,
            ),
            _contract(
                "opt_in_pickle_loads_checked_bytes",
                area,
                pkl_ok,
                "the bytes whose sha256 matched are the bytes deserialized",
            ),
            _contract(
                "probe_never_deserializes",
                area,
                probe_ok,
                "probe_artifact returns format/sha256/version metadata only — "
                "pickle bytes are hashed, never executed",
            ),
            _contract(
                "torch_weights_only_default",
                area,
                torch_ok,
                torch_detail,
            ),
            _flag(
                "safe_load_stamps_post_read_digest",
                area,
                "json/onnx/torch-weights-only paths deserialize first and hash "
                "the file afterwards — a mid-load rewrite could make the "
                "stamped digest describe different bytes (the runner re-probes "
                "after load; single-shot callers see the gap)",
            ),
            _flag(
                "sidecar_version_unhashed",
                area,
                "the <path>.version sidecar stamps metadata without "
                "digest-binding — a version string can outlive its payload",
            ),
        ]


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def _determinism_probes(prior_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    area = "determinism"
    repeat = [row for group in _PROBE_GROUPS for row in group()]
    return [
        _contract(
            "audit_results_reproducible",
            area,
            repeat == prior_rows,
            "a second independent probe pass produces identical result rows",
        ),
        _contract(
            "canonical_json_sorted_bytes",
            area,
            canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}',
            "receipt digest convention is byte-stable",
        ),
    ]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

_PROBE_GROUPS: tuple[Callable[[], list[dict[str, str]]], ...] = (
    _schema_feature_probes,
    _schema_forecast_probes,
    _visibility_probes,
    _causality_probes,
    _resample_probes,
    _artifact_probes,
)


def _run_group(group: Callable[[], list[dict[str, str]]]) -> list[dict[str, str]]:
    try:
        return group()
    except Exception as exc:  # a probe that escapes its own error handling
        return [
            {
                "probe": f"{group.__name__}_crashed",
                "area": "harness",
                "verdict": "fail",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        ]


def forecast_data_audit_results() -> list[dict[str, str]]:
    """Run every audit probe. A crashed probe group is recorded, never raised."""
    results: list[dict[str, str]] = []
    for group in _PROBE_GROUPS:
        results.extend(_run_group(group))
    try:
        results.extend(_determinism_probes(list(results)))
    except Exception as exc:
        results.append(
            {
                "probe": "_determinism_probes_crashed",
                "area": "harness",
                "verdict": "fail",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
    return results


def _interpretation(results: list[dict[str, str]]) -> str:
    counts = {verdict: 0 for verdict in ("pass", "fixed", "flag", "fail")}
    for row in results:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    return (
        f"fx1 forecast data layer adversarial audit: {counts['pass']} contract "
        f"probes pass, {counts['fixed']} defects fixed in this lane, "
        f"{counts['flag']} tolerated surfaces pinned as flags, "
        f"{counts['fail']} failures. Synthetic fixtures only — correctness "
        f"evidence, not market evidence."
    )


def forecast_data_audit() -> dict[str, Any]:
    """Run all probes and return the unsealed ``forecast_data_audit.v1`` body."""
    results = forecast_data_audit_results()
    ok = all(row["verdict"] in _PASSING_VERDICTS for row in results)
    return {
        "kind": AUDIT_KIND,
        "schema": AUDIT_SCHEMA,
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": results, "ok": ok},
        "interpretation": _interpretation(results),
    }


def forecast_data_audit_bench() -> dict[str, Any]:
    """Seal the audit body: digest over the canonical bytes, then stamped."""
    out = forecast_data_audit()
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


def write_forecast_data_audit_receipt(
    out_dir: Path | str = Path("receipts"),
) -> Path:
    """Write the sealed audit to ``<out_dir>/forecast_data_audit.json``."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    receipt = forecast_data_audit_bench()
    path = directory / AUDIT_RECEIPT_NAME
    atomic_write_text(path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path
