"""Adversarial audit lane for the fx-1 forecast core.

``forecast_core_audit`` probes the real behavior of ``runner``,
``evaluate``, and ``config`` (and the schema, features, signals, registry,
and artifact machinery they call) and returns ``probe name -> bool | str``.
Every bool ``True`` means the pinned contract held; string probes record an
observed constant such as a label or schema tag. ``flag_*`` probes pin
legitimate-but-surprising surfaces exactly as implemented — documented
limits, coarse checks, and conventions — rather than idealized behavior.

``forecast_core_audit_bench`` seals the probe map in a
``forecast_core_audit.v1`` receipt. Everything here runs on in-memory or
synthetic panels under a tmp dir: no real market data, no orders.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from pydantic import ValidationError

from fx1.forecast.artifacts import UntrustedArtifactError
from fx1.forecast.config import Fx1HarnessConfig
from fx1.forecast.evaluate import (
    _decision_times as _forecast_decision_times,
)
from fx1.forecast.evaluate import (
    _hit_rate,
    _mae_rmse,
    _score_column,
    _signal_diagnostics,
    _test_folds,
    evaluate_forecasts,
    forward_simple_returns,
    json_ready,
)
from fx1.forecast.features import (
    OhlcvFeaturePipeline,
    normalize_bar_times,
    resample_ohlcv,
    visible_bars,
)
from fx1.forecast.protocol import ForecastModel
from fx1.forecast.registry import ModelNotRegistered, create_model, load_symbol
from fx1.forecast.runner import (
    _META_KEY,
    data_label,
    load_bars,
    parse_bound,
    resolve_provider,
    run_inference,
    run_signal_evaluation,
)
from fx1.forecast.schema import (
    SchemaError,
    validate_feature_schema,
    validate_forecast_schema,
)
from fx1.forecast.signals import PLACEHOLDER_NOT_A_STRATEGY, map_signals
from fx1.honesty import FORBIDDEN_HEADLINE_TOKENS
from quant_fund.schemas.errors import LeakageError, PointInTimeError
from quant_fund.utils.hashing import (
    canonical_frame_fingerprint,
    canonical_json_bytes,
    hash_bytes,
    hash_file,
)
from quant_fund.utils.reproducibility import git_revision

ProbeValue = bool | str


def _mapping_keys(obj: object) -> list[str]:
    """Nested mapping keys (dicts only; list elements walked)."""
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(str(k))
            keys.extend(_mapping_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.extend(_mapping_keys(item))
    return keys


def _forbidden_metric_keys_absent(payload: object) -> bool:
    """fx1-side mirror of ``family_blob_forbidden_metrics_absent``.

    ``quant_fund.research.catalog`` sits outside fx1's allowed harness
    surface (configs/arch_boundaries.toml), so the same underscore-token
    check is re-run against ``FORBIDDEN_HEADLINE_TOKENS`` — kept identical
    by ``tests/fx1/test_honesty_inheritance.py``. ``live_pnl_claim`` stays
    exempt: it is the honesty flag, not a metric.
    """
    for key in _mapping_keys(payload):
        if key == "live_pnl_claim":
            continue
        parts = str(key).lower().replace("-", "_").split("_")
        if any(tok in FORBIDDEN_HEADLINE_TOKENS for tok in parts if tok):
            return False
    return True


def _stamp(offset: int) -> datetime:
    return datetime(2020, 1, 1, 16, tzinfo=UTC) + timedelta(days=offset)


def _panel(
    n: int = 24,
    names: tuple[str, ...] = ("AAA", "BBB", "CCC"),
    *,
    late: dict[int, int] | None = None,
    source: str = "synthetic",
) -> pl.DataFrame:
    """Deterministic daily bars. ``late`` maps a bar index to days of release delay."""
    rows: list[dict[str, object]] = []
    delays = late or {}
    for s_i, sid in enumerate(names):
        price = 40.0 + 5.0 * s_i
        for i in range(n):
            price *= 1.01 if (i + s_i) % 2 == 0 else 0.99
            event = _stamp(i)
            available = event + timedelta(days=delays.get(i, 0))
            rows.append(
                {
                    "security_id": sid,
                    "event_time": event,
                    "available_time": available,
                    "ingested_time": available,
                    "source": source,
                    "revision_id": "v1",
                    "open": price * 0.99,
                    "high": price * 1.02,
                    "low": price * 0.98,
                    "close": price,
                    "volume": 1_000.0 + i,
                    "planted_signal": 1.0 if i % 2 == 0 else -1.0,
                }
            )
    return pl.DataFrame(rows)


class _MemoryProvider:
    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame

    def get_bars(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        security_ids: list[str] | None = None,
    ) -> pl.DataFrame:
        frame = self.frame
        if start is not None:
            frame = frame.filter(pl.col("event_time") >= start)
        if end is not None:
            frame = frame.filter(pl.col("event_time") <= end)
        if security_ids:
            frame = frame.filter(pl.col("security_id").is_in(list(security_ids)))
        return frame


class _DictBarsProvider:
    """Provider-contract probe object: returns a dict, not a frame."""

    def get_bars(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        security_ids: list[str] | None = None,
    ) -> dict[str, str]:
        return {"not": "frame"}


class _NoGetBars:
    """Provider/model-contract probe object: no ``get_bars`` or ``predict``."""


class _SpyModel(ForecastModel):
    """Reference-conforming model that records what each predict call saw."""

    name = "audit-spy"
    version = "audit"

    def __init__(self) -> None:
        self.horizons: list[datetime] = []
        self.seen: list[set[datetime]] = []
        self.calls = 0
        self.horizon_bars = 1

    def load(
        self,
        checkpoint_path: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.horizon_bars = int((config or {}).get("horizon_bars", 1))

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        self.calls += 1
        self.horizons.append(cast(datetime, features["event_time"].max()))
        self.seen.append(set(features["event_time"].to_list()))
        n = features.height
        return pl.DataFrame(
            {
                "event_time": features["event_time"],
                "security_id": features["security_id"],
                "horizon_bars": pl.Series([self.horizon_bars] * n, dtype=pl.Int64),
                "predicted_return": pl.Series([0.0] * n, dtype=pl.Float64),
            }
        )


class _FutureModel(ForecastModel):
    """Predicts rows one day past the decision cutoff."""

    name = "audit-future"
    version = "audit"

    def load(
        self,
        checkpoint_path: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.horizon_bars = int((config or {}).get("horizon_bars", 1))

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        n = features.height
        future_times = [stamp + timedelta(days=1) for stamp in features["event_time"].to_list()]
        return pl.DataFrame(
            {
                "event_time": pl.Series(future_times, dtype=pl.Datetime("us", "UTC")),
                "security_id": features["security_id"],
                "horizon_bars": pl.Series([self.horizon_bars] * n, dtype=pl.Int64),
                "predicted_return": pl.Series([0.0] * n, dtype=pl.Float64),
            }
        )


class _DropOneModel(_SpyModel):
    """Drops the last feature row so the forecast misses a security."""

    name = "audit-drop"

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        if features.height <= 1:
            return super().predict(features)
        return super().predict(features.head(features.height - 1))


class _DictModel(ForecastModel):
    """Returns a dict instead of a frame from predict."""

    name = "audit-dict"
    version = "audit"

    def load(
        self,
        checkpoint_path: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        return None

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        return cast(pl.DataFrame, {"event_time": []})


class _EmptyModel(_SpyModel):
    name = "audit-empty"

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame()


class _NeverLoadModel(_SpyModel):
    name = "audit-neverload"

    def load(
        self,
        checkpoint_path: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        raise AssertionError("model.load must not run for an untrusted checkpoint")


class _MutateCheckpointModel(_SpyModel):
    """Appends a byte to the checkpoint file during load."""

    name = "audit-mutate"

    def load(
        self,
        checkpoint_path: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        path = Path(str(checkpoint_path))
        path.write_bytes(path.read_bytes() + b"\x00")


class _AuditOnlyModel(_SpyModel):
    """A ForecastModel that is not an Fx1Model, for entrypoint probes."""

    name = "audit-not-fx1"


class _CountingPipeline(OhlcvFeaturePipeline):
    """Counts ``build`` calls to pin the walk-forward build strategy."""

    builds: int = 0

    def __init__(self) -> None:
        super().__init__([1], 5)

    def build(self, bars: pl.DataFrame, *, decision_time: datetime | None = None) -> pl.DataFrame:
        _CountingPipeline.builds += 1
        return super().build(bars, decision_time=decision_time)


def _raised(
    kinds: type[BaseException] | tuple[type[BaseException], ...],
    fn: Callable[[], object],
) -> bool:
    """True iff ``fn`` raises one of ``kinds`` — a wrong-type error fails the probe."""
    expected = kinds if isinstance(kinds, tuple) else (kinds,)
    try:
        fn()
    except expected:
        return True
    except Exception:
        return False
    return False


def _ok(fn: Callable[[], object]) -> bool:
    """True iff ``fn`` returns without raising."""
    try:
        fn()
    except Exception:
        return False
    return True


def _attempt[T](fn: Callable[[], T]) -> T | bool:
    """Run one probe; a crash means the contract did not hold."""
    try:
        return fn()
    except Exception:
        return False


def _base_payload() -> dict[str, Any]:
    return {
        "schema": "fx1.harness.config/v1",
        "data": {
            "provider": "entrypoint",
            "entrypoint": "fx1.forecast.dummy:ZeroForecastModel",
        },
        "features": {"lookbacks": [1], "vol_window": 5, "horizon_bars": 1},
        "model": {"name": "dummy-zero"},
        "signal": {"mapping": "sign", "threshold": 0.0, "cost_bps": 5.0},
        "walk_forward": {
            "scheme": "expanding",
            "train_bars": 8,
            "val_bars": 3,
            "test_bars": 3,
            "embargo_bars": 1,
        },
    }


def _merge_sections(payload: dict[str, Any], sections: dict[str, Any]) -> None:
    for key, value in sections.items():
        section = payload.setdefault(key, {})
        assert isinstance(section, dict)
        section.update(value)


def _config(out_dir: Path, **overrides: Any) -> Fx1HarnessConfig:
    payload = _base_payload()
    payload["inference"] = {
        "mode": "walk_forward",
        "output_parquet": str(out_dir / "forecasts.parquet"),
        "output_meta": str(out_dir / "forecasts.meta.json"),
    }
    _merge_sections(payload, overrides)
    return Fx1HarnessConfig.model_validate(payload)


def _config_probes() -> dict[str, ProbeValue]:
    out: dict[str, ProbeValue] = {}

    def _validate(**sections: Any) -> Fx1HarnessConfig:
        payload = _base_payload()
        _merge_sections(payload, sections)
        return Fx1HarnessConfig.model_validate(payload)

    def _with(section: str, kw: dict[str, Any]) -> Fx1HarnessConfig:
        return _validate(**{section: kw})

    out["config_forbids_unknown_keys"] = _raised(
        ValidationError,
        lambda: Fx1HarnessConfig.model_validate({**_base_payload(), "surprise_section": {}}),
    )
    out["config_section_forbids_unknown_keys"] = _raised(
        ValidationError,
        lambda: _validate(data={"provider": "synthetic", "mystery": 1}),
    )
    out["config_parquet_requires_root"] = _raised(
        ValidationError, lambda: _validate(data={"provider": "parquet"})
    )
    out["config_entrypoint_requires_path"] = _raised(
        ValidationError,
        lambda: _validate(data={"provider": "entrypoint", "entrypoint": None}),
    )
    out["config_synthetic_bounds"] = all(
        _raised(ValidationError, partial(_with, "data", kw))
        for kw in (
            {"provider": "synthetic", "synthetic_n_assets": 1},
            {"provider": "synthetic", "synthetic_n_days": 15},
        )
    )
    out["config_lookbacks_positive_and_unique"] = all(
        _raised(ValidationError, partial(_with, "features", {"lookbacks": lb}))
        for lb in ([], [0], [-3], [2, 2])
    )
    out["config_vol_window_min"] = _raised(
        ValidationError, lambda: _validate(features={"vol_window": 1})
    )
    out["config_horizon_min"] = _raised(
        ValidationError, lambda: _validate(features={"horizon_bars": 0})
    )
    out["config_trusted_sha_pattern"] = all(
        _raised(
            ValidationError,
            partial(_with, "model", {"trusted_checkpoint_sha256": sha}),
        )
        for sha in ("A" * 64, "0" * 63, "g" * 64)
    ) and _ok(partial(_with, "model", {"trusted_checkpoint_sha256": "0" * 64}))
    out["config_unsafe_requires_trusted_sha"] = _raised(
        ValidationError,
        lambda: _validate(model={"allow_unsafe_deserialization": True}),
    )
    out["config_signal_rejects_negative"] = all(
        _raised(ValidationError, partial(_with, "signal", kw))
        for kw in ({"threshold": -0.1}, {"cost_bps": -1.0})
    )
    out["config_signal_rejects_nonfinite"] = all(
        _raised(ValidationError, partial(_with, "signal", kw))
        for kw in (
            {"threshold": float("nan")},
            {"threshold": float("inf")},
            {"cost_bps": float("nan")},
            {"cost_bps": float("inf")},
        )
    )
    out["config_inference_mode_literal"] = _raised(
        ValidationError, lambda: _validate(inference={"mode": "nightly"})
    )
    out["config_checkpoint_format_literal"] = _raised(
        ValidationError, lambda: _validate(model={"checkpoint_format": "zip"})
    )
    out["config_inference_outputs_must_differ"] = all(
        _raised(ValidationError, partial(_with, "inference", pair))
        for pair in (
            {"output_parquet": "same.parquet", "output_meta": "same.parquet"},
            {"output_parquet": "a/b.parquet", "output_meta": "a/./b.parquet"},
        )
    )
    out["config_walkforward_bounds"] = all(
        _raised(ValidationError, partial(_with, "walk_forward", section))
        for section in (
            {"train_bars": 0},
            {"val_bars": 0},
            {"test_bars": 0},
            {"embargo_bars": -1},
        )
    )
    out["config_schema_alias_parses"] = _ok(lambda: _validate())
    return out


def _schema_probes() -> dict[str, ProbeValue]:
    out: dict[str, ProbeValue] = {}
    t0 = _stamp(0)
    feats = pl.DataFrame(
        {
            "event_time": [t0, t0],
            "security_id": ["AAA", "BBB"],
            "close": [10.0, 20.0],
            "available_time": [t0, t0],
            "ret_1": [0.01, -0.01],
            "vol_5": [0.02, 0.03],
        }
    )
    columns = ["ret_1", "vol_5"]
    fc = pl.DataFrame(
        {
            "event_time": [t0],
            "security_id": ["AAA"],
            "horizon_bars": pl.Series([1], dtype=pl.Int64),
            "predicted_return": [0.05],
        }
    )

    out["feature_schema_accepts_clean_frame"] = _ok(lambda: validate_feature_schema(feats, columns))

    def _extra_feature_col(col: str) -> None:
        validate_feature_schema(feats.with_columns(pl.lit(0.5).alias(col)), columns)

    out["feature_schema_rejects_lookahead"] = all(
        _raised(LeakageError, partial(_extra_feature_col, col))
        for col in ("target_return", "realized_return", "fwd_ret", "label")
    )
    out["feature_schema_rejects_extra_column"] = _raised(
        SchemaError,
        lambda: validate_feature_schema(feats.with_columns(pl.lit(1.0).alias("mystery")), columns),
    )
    out["feature_schema_rejects_missing_column"] = _raised(
        SchemaError, lambda: validate_feature_schema(feats.drop("ret_1"), columns)
    )
    out["feature_schema_rejects_nonfinite"] = _raised(
        SchemaError,
        lambda: validate_feature_schema(
            feats.with_columns(pl.lit(float("inf")).alias("ret_1")), columns
        ),
    )
    out["feature_schema_rejects_duplicate_rows"] = _raised(
        SchemaError, lambda: validate_feature_schema(pl.concat([feats, feats]), columns)
    )
    out["forecast_schema_accepts_clean_frame"] = _ok(lambda: validate_forecast_schema(fc))
    out["forecast_schema_rejects_missing_key"] = _raised(
        SchemaError, lambda: validate_forecast_schema(fc.drop("security_id"))
    )
    out["forecast_schema_needs_a_prediction"] = _raised(
        SchemaError, lambda: validate_forecast_schema(fc.drop("predicted_return"))
    )
    out["forecast_schema_rejects_label_columns"] = _raised(
        LeakageError,
        lambda: validate_forecast_schema(fc.with_columns(pl.lit(0.2).alias("realized_return"))),
    )
    out["forecast_schema_rejects_bad_horizon"] = _raised(
        SchemaError,
        lambda: validate_forecast_schema(fc.with_columns(pl.lit(0).alias("horizon_bars"))),
    ) and _raised(
        SchemaError,
        lambda: validate_forecast_schema(fc.with_columns(pl.lit("one").alias("horizon_bars"))),
    )
    out["forecast_schema_rejects_duplicate_keys"] = _raised(
        SchemaError, lambda: validate_forecast_schema(pl.concat([fc, fc]))
    )
    out["forecast_schema_rejects_unordered_quantiles"] = _raised(
        SchemaError,
        lambda: validate_forecast_schema(
            fc.with_columns(pl.lit(0.9).alias("q_0.1"), pl.lit(0.1).alias("q_0.9"))
        ),
    )

    def _with_confidence(value: float) -> None:
        validate_forecast_schema(fc.with_columns(pl.lit(value).alias("confidence")))

    out["forecast_schema_rejects_bad_confidence"] = all(
        _raised(SchemaError, partial(_with_confidence, value))
        for value in (1.5, -0.1, float("nan"))
    )

    def _with_prediction(update: dict[str, pl.Expr]) -> None:
        validate_forecast_schema(fc.with_columns(**update))

    out["forecast_schema_rejects_bad_predictions"] = all(
        _raised(SchemaError, partial(_with_prediction, update))
        for update in (
            {"predicted_return": pl.lit(float("inf"))},
            {"predicted_price": pl.lit(-1.0)},
            {"predicted_price": pl.lit(float("nan"))},
        )
    )
    # Documented coarse check: a row with any null quantile is skipped by the
    # ordering check, so a partial-null row can hide an inversion.
    out["flag_quantile_order_skips_partial_null_rows"] = _ok(
        lambda: validate_forecast_schema(
            fc.with_columns(
                pl.lit(0.9).alias("q_0.1"),
                pl.Series("q_0.5", [None], dtype=pl.Float64),
                pl.lit(0.1).alias("q_0.9"),
            )
        )
    )
    return out


def _feature_probes() -> dict[str, ProbeValue]:
    out: dict[str, ProbeValue] = {}
    pipeline = OhlcvFeaturePipeline([1, 5], 5)

    out["visible_bars_requires_available_time"] = _raised(
        PointInTimeError,
        lambda: visible_bars(_panel(6).drop("available_time"), _stamp(3)),
    )
    out["visible_bars_rejects_event_after_release"] = _raised(
        PointInTimeError,
        lambda: visible_bars(
            _panel(6).with_columns(
                pl.when(pl.col("event_time") == _stamp(2))
                .then(pl.lit(_stamp(1)))
                .otherwise(pl.col("available_time"))
                .alias("available_time")
            ),
            _stamp(5),
        ),
    )

    def _late_release_hidden() -> bool:
        bars = _panel(16, ("AAA",), late={10: 4})
        before = pipeline.build(bars, decision_time=_stamp(12))
        after = pipeline.build(bars, decision_time=_stamp(15))
        late_event = _stamp(10)
        return late_event not in set(before["event_time"].to_list()) and late_event in set(
            after["event_time"].to_list()
        )

    out["features_hide_bars_until_release"] = _attempt(_late_release_hidden)
    out["features_reject_duplicate_bars"] = _raised(
        PointInTimeError,
        lambda: OhlcvFeaturePipeline([1], 5).build(
            pl.concat([_panel(8, ("AAA",)), _panel(8, ("AAA",)).head(1)])
        ),
    )
    out["features_require_close"] = _raised(
        PointInTimeError,
        lambda: OhlcvFeaturePipeline([1], 5).build(_panel(8).drop("close")),
    )

    def _ret1_is_trailing() -> bool:
        bars = _panel(8, ("AAA",))
        built = OhlcvFeaturePipeline([1, 5], 5).build(bars).sort("event_time")
        closes = bars.sort("event_time")["close"].to_list()
        last = built.row(-1, named=True)
        i = len(closes) - 1
        expected_ret = closes[i] / closes[i - 1] - 1.0
        expected_mom = closes[i] / closes[i - 5] - 1.0
        return math.isclose(float(last["ret_1"]), expected_ret, rel_tol=1e-9) and math.isclose(
            float(last["mom_5"]), expected_mom, rel_tol=1e-9
        )

    out["feature_returns_are_trailing"] = _attempt(_ret1_is_trailing)
    out["features_drop_label_columns"] = (
        "planted_signal" not in pipeline.build(_panel(12, ("AAA",))).columns
    )

    def _normalize_naive_means_utc() -> bool:
        naive = _panel(4, ("AAA",)).with_columns(pl.col("event_time").dt.replace_time_zone(None))
        normalized = normalize_bar_times(naive)
        dtype = normalized.schema["event_time"]
        return isinstance(dtype, pl.Datetime) and dtype.time_zone == "UTC"

    out["flag_naive_datetimes_assumed_utc"] = _attempt(_normalize_naive_means_utc)
    out["normalize_rejects_string_times"] = _raised(
        PointInTimeError,
        lambda: normalize_bar_times(
            _panel(4, ("AAA",)).with_columns(pl.lit("2020-01-01").alias("event_time"))
        ),
    )

    def _resample_labels_last_print() -> bool:
        day = _stamp(0).date()
        rows = []
        for hour, close in ((10, 10.0), (14, 12.0)):
            stamp = datetime(day.year, day.month, day.day, hour, tzinfo=UTC)
            rows.append(
                {
                    "security_id": "AAA",
                    "event_time": stamp,
                    "available_time": stamp,
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 5.0,
                    "source": "synthetic",
                }
            )
        nxt = day + timedelta(days=1)
        later = datetime(nxt.year, nxt.month, nxt.day, 11, tzinfo=UTC)
        rows.append(
            {
                "security_id": "AAA",
                "event_time": later,
                "available_time": later,
                "open": 99.0,
                "high": 100.0,
                "low": 98.0,
                "close": 99.0,
                "volume": 7.0,
                "source": "synthetic",
            }
        )
        resampled = resample_ohlcv(pl.DataFrame(rows), "1d").sort("event_time")
        first = resampled.row(0, named=True)
        return bool(
            first["event_time"] == _stamp(0).replace(hour=14)
            and first["close"] == 12.0
            and first["volume"] == 10.0
            and resampled.row(1, named=True)["close"] == 99.0
        )

    out["resample_bucket_labeled_by_last_print"] = _attempt(_resample_labels_last_print)
    out["resample_rejects_mixed_sources"] = _raised(
        PointInTimeError,
        lambda: resample_ohlcv(
            pl.concat(
                [
                    _panel(2, ("AAA",), source="a"),
                    _panel(2, ("AAA",), source="b"),
                ]
            ),
            "1d",
        ),
    )

    def _resample_rejects_unreleased_last_print() -> bool:
        bars = _panel(2, ("AAA",)).with_columns(
            pl.when(pl.col("event_time") == _stamp(1))
            .then(pl.lit(_stamp(1) - timedelta(hours=2)))
            .otherwise(pl.col("available_time"))
            .alias("available_time")
        )
        return _raised(PointInTimeError, lambda: resample_ohlcv(bars, "1d"))

    out["resample_rejects_unreleased_last_print"] = _attempt(
        _resample_rejects_unreleased_last_print
    )
    out["resample_rejects_empty_interval"] = _raised(
        ValueError, lambda: resample_ohlcv(_panel(2, ("AAA",)), "  ")
    )
    out["pipeline_rejects_bad_windows"] = all(
        _raised(ValueError, partial(OhlcvFeaturePipeline, lb, vw))
        for lb, vw in (([], 5), ([0], 5), ([2, 2], 5), ([1], 1))
    )
    return out


def _registry_probes() -> dict[str, ProbeValue]:
    out: dict[str, ProbeValue] = {}
    out["registry_fx1_name_fails_closed"] = all(
        _raised(ModelNotRegistered, partial(create_model, name)) for name in ("fx-1", "fx1")
    )
    out["registry_fx1_entrypoint_must_subclass_fx1model"] = _raised(
        TypeError,
        lambda: create_model("fx-1", entrypoint="fx1.forecast.core_audit:_AuditOnlyModel"),
    )
    out["registry_entrypoint_must_be_forecast_model"] = _raised(
        TypeError,
        lambda: create_model("other", entrypoint="fx1.forecast.core_audit:_NoGetBars"),
    )
    out["load_symbol_requires_module_colon_attr"] = all(
        _raised(ValueError, partial(load_symbol, spec))
        for spec in ("nocolon", ":attr", "mod:", ":")
    )
    out["load_symbol_missing_attr_fails_closed"] = _raised(
        ModelNotRegistered, lambda: load_symbol("fx1.forecast.dummy:NoSuchModel")
    )
    out["load_symbol_missing_module_propagates_import_error"] = _raised(
        ImportError, lambda: load_symbol("no.such.module:Thing")
    )

    def _provider_entrypoint_contract() -> bool:
        config = _config(
            Path(tempfile.mkdtemp()),
            data={
                "provider": "entrypoint",
                "entrypoint": "fx1.forecast.core_audit:_NoGetBars",
            },
        )
        return _raised(TypeError, lambda: resolve_provider(config))

    out["provider_entrypoint_requires_get_bars"] = _attempt(_provider_entrypoint_contract)
    return out


def _runner_probes(tmp: Path) -> dict[str, ProbeValue]:
    out: dict[str, ProbeValue] = {}
    bars = _panel(24)

    out["parse_bound_z_and_naive_to_utc"] = _attempt(
        lambda: (
            parse_bound("2020-01-02T00:00:00Z") == datetime(2020, 1, 2, tzinfo=UTC)
            and parse_bound("2020-01-02 00:00:00") == datetime(2020, 1, 2, tzinfo=UTC)
            and parse_bound(None) is None
            and parse_bound("   ") is None
        )
    )
    out["parse_bound_garbage_fails"] = _raised(ValueError, lambda: parse_bound("garbage"))
    out["provider_bars_missing_pit_columns_fail"] = all(
        _raised(
            PointInTimeError,
            partial(
                load_bars,
                _MemoryProvider(_panel(4).drop(col)),
                _config(tmp / f"nopit-{col}"),
            ),
        )
        for col in ("event_time", "available_time")
    )
    out["provider_empty_bars_fail"] = _raised(
        ValueError,
        lambda: load_bars(_MemoryProvider(pl.DataFrame()), _config(tmp / "empty")),
    )
    out["provider_non_frame_fails"] = _raised(
        TypeError,
        lambda: load_bars(_DictBarsProvider(), _config(tmp / "nonframe")),
    )
    out["symbol_filter_empty_fails"] = _raised(
        ValueError,
        lambda: run_inference(
            _config(tmp / "nosyms", data={"symbols": ["ZZZ"]}),
            model=_SpyModel(),
            provider=_MemoryProvider(bars),
        ),
    )
    out["empty_decision_window_fails"] = _raised(
        ValueError,
        lambda: run_inference(
            _config(tmp / "nodes", data={"start": "2030-01-01"}),
            model=_SpyModel(),
            provider=_MemoryProvider(bars),
        ),
    )

    def _walk_forward_truncates() -> bool:
        spy = _SpyModel()
        result = run_inference(_config(tmp / "wf"), model=spy, provider=_MemoryProvider(bars))
        if not spy.horizons or spy.horizons != sorted(spy.horizons):
            return False
        if set(spy.horizons) != set(result.forecasts["event_time"].to_list()):
            return False
        return all(
            max(seen) == horizon for horizon, seen in zip(spy.horizons, spy.seen, strict=True)
        )

    out["walk_forward_truncates_at_decision"] = _attempt(_walk_forward_truncates)

    def _late_release_hidden_from_model() -> bool:
        spy = _SpyModel()
        late_bars = _panel(20, ("AAA", "BBB"), late={10: 4})
        run_inference(_config(tmp / "late"), model=spy, provider=_MemoryProvider(late_bars))
        late_event, release = _stamp(10), _stamp(14)
        return all(
            late_event not in seen
            for horizon, seen in zip(spy.horizons, spy.seen, strict=True)
            if horizon < release
        ) and any(
            late_event in seen
            for horizon, seen in zip(spy.horizons, spy.seen, strict=True)
            if horizon >= release
        )

    out["walk_forward_hides_late_release"] = _attempt(_late_release_hidden_from_model)
    out["batch_refuses_late_release"] = _raised(
        LeakageError,
        lambda: run_inference(
            _config(
                tmp / "batchlate",
                inference={
                    "mode": "batch",
                    "output_parquet": str(tmp / "batchlate" / "f.parquet"),
                    "output_meta": str(tmp / "batchlate" / "f.meta.json"),
                },
            ),
            model=_SpyModel(),
            provider=_MemoryProvider(_panel(20, late={10: 4})),
        ),
    )
    out["forecast_future_event_time_fails"] = _raised(
        LeakageError,
        lambda: run_inference(
            _config(tmp / "fut"),
            model=_FutureModel(),
            provider=_MemoryProvider(bars),
        ),
    )
    out["forecast_wrong_cross_section_fails"] = _raised(
        SchemaError,
        lambda: run_inference(
            _config(tmp / "drop"),
            model=_DropOneModel(),
            provider=_MemoryProvider(bars),
        ),
    )
    out["forecast_non_frame_fails"] = _raised(
        TypeError,
        lambda: run_inference(
            _config(tmp / "dict"),
            model=_DictModel(),
            provider=_MemoryProvider(bars),
        ),
    )
    out["forecast_empty_frame_fails"] = _raised(
        SchemaError,
        lambda: run_inference(
            _config(tmp / "emptyfc"),
            model=_EmptyModel(),
            provider=_MemoryProvider(bars),
        ),
    )
    out["batch_forecast_key_mismatch_fails"] = _raised(
        SchemaError,
        lambda: run_inference(
            _config(
                tmp / "bmis",
                inference={
                    "mode": "batch",
                    "output_parquet": str(tmp / "bmis" / "f.parquet"),
                    "output_meta": str(tmp / "bmis" / "f.meta.json"),
                },
            ),
            model=_DropOneModel(),
            provider=_MemoryProvider(bars),
        ),
    )

    def _pickle_needs_trust() -> bool:
        path = tmp / "candidate.pkl"
        path.write_bytes(b"checkpoint bytes")
        config = _config(
            tmp / "pkl",
            model={"checkpoint_path": str(path), "checkpoint_format": "pickle"},
        )
        return _raised(
            UntrustedArtifactError,
            lambda: run_inference(config, model=_NeverLoadModel(), provider=_MemoryProvider(bars)),
        )

    out["checkpoint_pickle_requires_trust"] = _attempt(_pickle_needs_trust)

    def _sha_mismatch() -> bool:
        path = tmp / "model.json"
        path.write_text(json.dumps({"horizon_bars": 1}), encoding="utf-8")
        config = _config(
            tmp / "shabad",
            model={
                "checkpoint_path": str(path),
                "checkpoint_format": "json",
                "trusted_checkpoint_sha256": "0" * 64,
            },
        )
        return _raised(
            UntrustedArtifactError,
            lambda: run_inference(config, model=_SpyModel(), provider=_MemoryProvider(bars)),
        )

    out["checkpoint_sha_mismatch_fails"] = _attempt(_sha_mismatch)

    def _mutation_caught() -> bool:
        path = tmp / "mut.json"
        path.write_text(json.dumps({"horizon_bars": 1}), encoding="utf-8")
        config = _config(
            tmp / "mut",
            model={"checkpoint_path": str(path), "checkpoint_format": "json"},
        )
        return _raised(
            UntrustedArtifactError,
            lambda: run_inference(
                config,
                model=_MutateCheckpointModel(),
                provider=_MemoryProvider(bars),
            ),
        )

    out["checkpoint_mutation_during_load_fails"] = _attempt(_mutation_caught)
    out["checkpoint_missing_file_fails"] = _raised(
        FileNotFoundError,
        lambda: run_inference(
            _config(
                tmp / "gone",
                model={"checkpoint_path": str(tmp / "gone" / "nope.json")},
            ),
            model=_SpyModel(),
            provider=_MemoryProvider(bars),
        ),
    )

    def _json_checkpoint_is_safe() -> bool:
        path = tmp / "safe.json"
        path.write_text(json.dumps({"horizon_bars": 1}), encoding="utf-8")
        config = _config(
            tmp / "safe",
            model={"checkpoint_path": str(path), "checkpoint_format": "json"},
        )
        result = run_inference(config, model=None, provider=_MemoryProvider(bars))
        return bool(result.metadata["artifact_sha256"] == hash_file(path))

    out["flag_json_checkpoint_loads_without_opt_in"] = _attempt(_json_checkpoint_is_safe)

    def _symbol_filter_restricts() -> bool:
        config = _config(tmp / "symfilter", data={"symbols": ["AAA"]})
        result = run_inference(config, model=_SpyModel(), provider=_MemoryProvider(bars))
        return set(result.forecasts["security_id"].to_list()) == {"AAA"}

    out["symbol_filter_restricts_forecasts"] = _attempt(_symbol_filter_restricts)

    def _start_filters_decisions_not_history() -> bool:
        spy = _SpyModel()
        start = _stamp(12)
        config = _config(tmp / "startbound", data={"start": start.isoformat()})
        result = run_inference(config, model=spy, provider=_MemoryProvider(bars))
        times = set(result.forecasts["event_time"].to_list())
        if not times or not all(t >= start for t in times):
            return False
        # Features at the first decision still see history before data.start.
        return any(min(seen) < start for seen in spy.seen if seen)

    out["flag_start_filters_decisions_not_history"] = _attempt(_start_filters_decisions_not_history)

    def _end_filters_bars() -> bool:
        end = _stamp(10)
        config = _config(tmp / "endbound", data={"end": end.isoformat()})
        result = run_inference(config, model=_SpyModel(), provider=_MemoryProvider(bars))
        times = set(result.forecasts["event_time"].to_list())
        return bool(times) and all(t <= end for t in times)

    out["flag_end_filters_bars_and_decisions"] = _attempt(_end_filters_bars)

    def _empty_decisions_skipped() -> bool:
        single = _panel(20, ("AAA",))
        spy = _SpyModel()
        result = run_inference(_config(tmp / "skip"), model=spy, provider=_MemoryProvider(single))
        bar_times = set(single["event_time"].to_list())
        forecast_times = set(result.forecasts["event_time"].to_list())
        skipped = bar_times - forecast_times
        return bool(skipped) and min(forecast_times) == _stamp(5)

    out["flag_walkforward_skips_empty_decisions"] = _attempt(_empty_decisions_skipped)

    def _punctual_builds_once() -> bool:
        _CountingPipeline.builds = 0
        config = _config(
            tmp / "once",
            features={"entrypoint": "fx1.forecast.core_audit:_CountingPipeline"},
        )
        run_inference(config, model=_SpyModel(), provider=_MemoryProvider(bars))
        return _CountingPipeline.builds == 1

    out["flag_punctual_bars_build_features_once"] = _attempt(_punctual_builds_once)

    def _late_release_rebuilds() -> bool:
        _CountingPipeline.builds = 0
        late_bars = _panel(16, ("AAA", "BBB"), late={10: 3})
        config = _config(
            tmp / "rebuild",
            features={"entrypoint": "fx1.forecast.core_audit:_CountingPipeline"},
        )
        run_inference(config, model=_SpyModel(), provider=_MemoryProvider(late_bars))
        return _CountingPipeline.builds == late_bars["event_time"].n_unique()

    out["flag_late_release_rebuilds_per_decision"] = _attempt(_late_release_rebuilds)

    def _batch_sees_whole_window() -> bool:
        spy = _SpyModel()
        run_inference(
            _config(
                tmp / "batch",
                inference={
                    "mode": "batch",
                    "output_parquet": str(tmp / "batch" / "f.parquet"),
                    "output_meta": str(tmp / "batch" / "f.meta.json"),
                },
            ),
            model=spy,
            provider=_MemoryProvider(bars),
        )
        return spy.calls == 1 and max(spy.seen[0]) == max(bars["event_time"].to_list())

    out["flag_batch_predict_sees_whole_window"] = _attempt(_batch_sees_whole_window)

    def _metadata_binds() -> bool:
        config = _config(tmp / "meta")
        result = run_inference(config, model=_SpyModel(), provider=_MemoryProvider(bars))
        expected = hash_bytes(canonical_json_bytes(config.model_dump(mode="json", by_alias=True)))
        meta = result.metadata
        return (
            meta["config_sha256"] == expected
            and meta["research_only"] is True
            and meta["live_pnl_claim"] is False
            and meta["model_role"] == "external"
            and meta["schema"] == "fx1.harness.forecasts/v1"
            and meta["n_rows"] == result.forecasts.height
        )

    out["metadata_binds_config_sha256"] = _attempt(_metadata_binds)

    def _parquet_embeds_metadata() -> bool:
        config = _config(tmp / "embed")
        result = run_inference(config, model=_SpyModel(), provider=_MemoryProvider(bars))
        schema_meta = pq.read_schema(result.parquet_path).metadata or {}
        embedded = json.loads(schema_meta[_META_KEY].decode("utf-8"))
        on_disk = json.loads(result.meta_path.read_text(encoding="utf-8"))
        return bool(embedded == on_disk and embedded["run_id"] == result.metadata["run_id"])

    out["parquet_embeds_meta_payload"] = _attempt(_parquet_embeds_metadata)

    def _deterministic_runs() -> bool:
        cfg_a = _config(tmp / "det-a")
        cfg_b = _config(tmp / "det-b")
        first = run_inference(cfg_a, model=_SpyModel(), provider=_MemoryProvider(bars))
        second = run_inference(cfg_b, model=_SpyModel(), provider=_MemoryProvider(bars))
        return bool(
            canonical_frame_fingerprint(first.forecasts)
            == canonical_frame_fingerprint(second.forecasts)
        )

    out["inference_deterministic_under_seed"] = _attempt(_deterministic_runs)

    def _run_ids_differ() -> bool:
        cfg_a = _config(tmp / "rid-a")
        cfg_b = _config(tmp / "rid-b")
        first = run_inference(cfg_a, model=_SpyModel(), provider=_MemoryProvider(bars))
        second = run_inference(cfg_b, model=_SpyModel(), provider=_MemoryProvider(bars))
        return bool(first.metadata["run_id"] != second.metadata["run_id"])

    out["flag_run_id_unique_per_run"] = _attempt(_run_ids_differ)

    def _config_sha_changes() -> bool:
        cfg_a = _config(tmp / "sha-a")
        cfg_b = _config(tmp / "sha-b", signal={"cost_bps": 9.0})
        res_a = run_inference(cfg_a, model=_SpyModel(), provider=_MemoryProvider(bars))
        res_b = run_inference(cfg_b, model=_SpyModel(), provider=_MemoryProvider(bars))
        return bool(res_a.metadata["config_sha256"] != res_b.metadata["config_sha256"])

    out["config_sha_binds_whole_config"] = _attempt(_config_sha_changes)

    out["data_label_synthetic_source"] = data_label(_panel(4, source="synthetic"), "parquet")
    mixed = pl.concat([_panel(4, ("AAA",), source="a"), _panel(4, ("BBB",), source="b")])
    out["flag_data_label_mixed_sources"] = data_label(mixed, "parquet")
    out["data_label_single_source_name"] = data_label(_panel(4, source="fixture"), "parquet")
    no_source = _panel(4).drop("source")
    out["data_label_provider_fallback"] = data_label(no_source, "synthetic")
    out["flag_data_label_passthrough"] = data_label(no_source, "parquet")

    out["eval_missing_parquet_fails"] = _raised(
        FileNotFoundError,
        lambda: run_signal_evaluation(_config(tmp / "noparq"), provider=_MemoryProvider(bars)),
    )

    def _synthetic_seed() -> bool:
        from quant_fund.data.adapters.synthetic import SyntheticMarketProvider

        first = canonical_frame_fingerprint(
            SyntheticMarketProvider(n_assets=3, n_days=32, seed=11).get_bars()
        )
        same = canonical_frame_fingerprint(
            SyntheticMarketProvider(n_assets=3, n_days=32, seed=11).get_bars()
        )
        other = canonical_frame_fingerprint(
            SyntheticMarketProvider(n_assets=3, n_days=32, seed=12).get_bars()
        )
        return first == same and first != other

    out["synthetic_seed_binds_output"] = _attempt(_synthetic_seed)
    out["canonical_json_key_order_invariant"] = hash_bytes(
        canonical_json_bytes({"a": 1, "b": {"x": [1, 2]}, "c": "z"})
    ) == hash_bytes(canonical_json_bytes({"c": "z", "b": {"x": [1, 2]}, "a": 1}))
    return out


def _eval_probes(tmp: Path) -> dict[str, ProbeValue]:
    out: dict[str, ProbeValue] = {}
    bars = _panel(48)

    def _forward_label() -> bool:
        labels = forward_simple_returns(bars, 2).sort(["security_id", "event_time"])
        closes = bars.filter(pl.col("security_id") == "AAA").sort("event_time")["close"]
        row = labels.filter(
            (pl.col("security_id") == "AAA") & (pl.col("event_time") == _stamp(3))
        ).row(0, named=True)
        return math.isclose(
            row["realized_return"], closes[5] / closes[3] - 1.0, rel_tol=1e-9
        ) and row["target_time"] == _stamp(5)

    out["forward_label_is_close_t_plus_h"] = _attempt(_forward_label)
    out["forward_horizon_minimum"] = _raised(ValueError, lambda: forward_simple_returns(bars, 0))
    out["json_ready_replaces_nonfinite"] = _attempt(
        lambda: (
            json_ready(
                {
                    "a": float("inf"),
                    "b": [float("nan")],
                    "c": np.float64(-np.inf),
                    "d": np.int64(3),
                }
            )
            == {"a": None, "b": [None], "c": None, "d": 3}
        )
    )
    out["score_column_prefers_predicted_return"] = _attempt(
        lambda: (
            _score_column(
                pl.DataFrame(
                    {
                        "predicted_return": [0.5],
                        "predicted_price": [200.0],
                        "close": [100.0],
                    }
                )
            )["score"].to_list()
            == [0.5]
        )
    ) and _attempt(
        lambda: math.isclose(
            _score_column(pl.DataFrame({"predicted_price": [200.0], "close": [100.0]}))[
                "score"
            ].to_list()[0],
            1.0,
            rel_tol=1e-9,
        )
    )

    def _mae_rmse_honest() -> bool:
        mae, rmse, n = _mae_rmse(np.array([0.1, np.nan, 0.3]), np.array([0.2, 0.2, np.nan]))
        return (
            n == 1
            and math.isclose(mae, 0.1, rel_tol=1e-9)
            and math.isclose(rmse, 0.1, rel_tol=1e-9)
        )

    out["mae_rmse_counts_finite_pairs_only"] = _attempt(_mae_rmse_honest)

    def _hit_rate_mask() -> bool:
        result = _hit_rate(np.array([0.5, 0.0, -0.5, 0.5]), np.array([0.5, 0.5, -0.5, 0.0]))
        pt = result["pt_stat"]
        return (
            result["hit_rate"] == 1.0
            and result["n_hit"] == 2
            and (pt is None or not math.isfinite(float(pt)))
        )

    out["flag_hit_rate_excludes_zero_signs"] = _attempt(_hit_rate_mask)

    def _pt_min_sample() -> bool:
        result = _hit_rate(np.ones(19), np.ones(19))
        pt = result["pt_stat"]
        return result["n_hit"] == 19 and (pt is None or not math.isfinite(float(pt)))

    out["flag_pt_needs_20_pairs"] = _attempt(_pt_min_sample)
    out["flag_pt_degenerate_margin_fails_closed"] = _raised(
        ValueError, lambda: _hit_rate(np.ones(25), np.ones(25))
    )

    def _signal_maps_bounded() -> bool:
        t0 = _stamp(0)
        frame = pl.DataFrame(
            {
                "event_time": [t0, t0, t0],
                "security_id": ["A", "B", "C"],
                "score": [0.2, -0.1, 0.0],
            }
        )
        signed = map_signals(frame, "sign").sort("security_id")["signal"].to_list()
        if signed != [1.0, -1.0, 0.0]:
            return False
        gated = map_signals(frame, "threshold", threshold=0.15).sort("security_id")
        if gated["signal"].to_list() != [1.0, 0.0, 0.0]:
            return False
        ranked = map_signals(frame, "rank")
        signal = ranked["signal"]
        lo = cast(float, signal.min())
        hi = cast(float, signal.max())
        return bool(
            lo >= -1.0
            and hi <= 1.0
            and set(ranked["mapping_role"].to_list()) == {PLACEHOLDER_NOT_A_STRATEGY}
        )

    out["signal_maps_bounded_and_marked"] = _attempt(_signal_maps_bounded)
    out["signal_map_unknown_fails"] = _raised(
        ValueError,
        lambda: map_signals(
            pl.DataFrame({"score": [0.1], "security_id": ["A"], "event_time": [_stamp(0)]}),
            "bogus",
        ),
    )
    score_frame = pl.DataFrame({"score": [0.1], "security_id": ["A"], "event_time": [_stamp(0)]})
    out["signal_map_rejects_nonfinite_threshold"] = all(
        _raised(
            ValueError,
            partial(map_signals, score_frame, "threshold", threshold=th),
        )
        for th in (float("inf"), float("-inf"), float("nan"))
    )
    out["signal_map_requires_score"] = _raised(
        SchemaError,
        lambda: map_signals(pl.DataFrame({"security_id": ["A"]}), "sign"),
    )

    def _rank_single_name() -> bool:
        out_frame = map_signals(
            pl.DataFrame({"event_time": [_stamp(0)], "security_id": ["A"], "score": [0.9]}),
            "rank",
        )
        return out_frame["signal"].to_list() == [0.0]

    out["flag_rank_single_name_maps_zero"] = _attempt(_rank_single_name)

    def _nonfinite_signals_skipped() -> bool:
        t0 = _stamp(0)
        frame = pl.DataFrame(
            {
                "event_time": [t0, t0, t0],
                "security_id": ["A", "B", "C"],
                "signal": [1.0, None, float("inf")],
            }
        )
        diag = _signal_diagnostics(frame, cost_bps=5.0, stride=1)
        return math.isclose(diag["mean_turnover"], 1.0, rel_tol=1e-9) and math.isclose(
            diag["mean_one_way_cost_drag_bps"], 5.0, rel_tol=1e-9
        )

    out["flag_null_and_nonfinite_signals_skipped"] = _attempt(_nonfinite_signals_skipped)

    def _two_sided_turnover() -> bool:
        t0, t1, t2 = _stamp(0), _stamp(1), _stamp(2)
        frame = pl.DataFrame(
            {
                "event_time": [t0, t1, t2],
                "security_id": ["A", "B", "B"],
                "signal": [1.0, 1.0, 1.0],
            }
        )
        diag = _signal_diagnostics(frame, cost_bps=5.0, stride=1)
        # sum|dw| convention: enter A = 1.0, flip A->B = 2.0, hold = 0.0.
        return math.isclose(diag["mean_turnover"], 1.0, rel_tol=1e-9)

    out["flag_turnover_is_two_sided"] = _attempt(_two_sided_turnover)

    out["eval_rejects_invalid_forecast_frame"] = _raised(
        SchemaError,
        lambda: evaluate_forecasts(
            pl.DataFrame({"event_time": [_stamp(0)], "security_id": ["A"]}),
            bars,
            _config(tmp / "evalbad"),
        ),
    )

    def _eval_no_folds_fails() -> bool:
        cfg = _config(
            tmp / "nofolds",
            walk_forward={
                "scheme": "expanding",
                "train_bars": 500,
                "val_bars": 3,
                "test_bars": 3,
                "embargo_bars": 1,
            },
        )
        result = run_inference(cfg, model=_SpyModel(), provider=_MemoryProvider(bars))
        return _raised(ValueError, lambda: evaluate_forecasts(result.forecasts, bars, cfg))

    out["eval_requires_test_folds"] = _attempt(_eval_no_folds_fails)

    def _eval_horizon_mismatch() -> bool:
        frame = pl.DataFrame(
            {
                "event_time": [_stamp(0)],
                "security_id": ["A"],
                "horizon_bars": pl.Series([2], dtype=pl.Int64),
                "predicted_return": [0.01],
            }
        )
        return _raised(
            SchemaError,
            lambda: evaluate_forecasts(frame, bars, _config(tmp / "horbad")),
        )

    out["eval_horizon_mismatch_fails"] = _attempt(_eval_horizon_mismatch)

    def _synthetic_cfg(tag: str) -> Fx1HarnessConfig:
        return _config(
            tmp / tag,
            data={
                "provider": "synthetic",
                "entrypoint": None,
                "synthetic_n_assets": 3,
                "synthetic_n_days": 48,
                "synthetic_seed": 11,
            },
        )

    e2e = _attempt(
        lambda: (
            run_signal_evaluation(
                _synthetic_cfg("e2e"),
                forecasts=run_inference(_synthetic_cfg("e2e")).forecasts,
            ),
            load_bars(resolve_provider(_synthetic_cfg("e2e")), _synthetic_cfg("e2e")),
        )
    )
    if isinstance(e2e, tuple):
        report, eval_bars = e2e
    else:
        report, eval_bars = {}, pl.DataFrame()
        out["_e2e_failed"] = False

    def _honesty() -> bool:
        return (
            report.get("schema") == "fx1.harness.eval/v1"
            and report.get("research_only") is True
            and report.get("live_pnl_claim") is False
            and report.get("promotion_eligible") is False
            and report.get("orders_submitted") is False
            and report.get("placeholder_signal_mapping") is True
            and report.get("mapping_role") == PLACEHOLDER_NOT_A_STRATEGY
            and report.get("data_label") == "SYNTHETIC"
        )

    out["eval_report_honesty_flags"] = _attempt(_honesty)
    out["eval_report_schema"] = cast(str, report.get("schema") or "")
    out["eval_report_no_forbidden_metric_keys"] = _attempt(
        lambda: bool(report) and _forbidden_metric_keys_absent(report)
    )

    def _scored_on_test_folds_only() -> bool:
        if not report or eval_bars.is_empty():
            return False
        result = run_inference(_synthetic_cfg("e2e2"))
        times = _forecast_decision_times(result.forecasts.filter(pl.col("horizon_bars") == 1))
        test_times, _folds, _info = _test_folds(times, _synthetic_cfg("e2e3"))
        joined = result.forecasts.filter(pl.col("horizon_bars") == 1).join(
            forward_simple_returns(eval_bars, 1),
            on=["security_id", "event_time"],
            how="left",
        )
        expected = joined.filter(
            pl.col("event_time").is_in(list(test_times)) & pl.col("realized_return").is_not_null()
        ).height
        by_h = report["forecast_metrics"]["by_horizon"]["1"]
        return by_h["n"] == expected and expected > 0

    out["eval_metrics_scored_on_test_folds_only"] = _attempt(_scored_on_test_folds_only)

    def _cost_drag_formula() -> bool:
        diag = report.get("signal_diagnostics") or {}
        drag = diag.get("mean_one_way_cost_drag_bps")
        cost = diag.get("cost_bps")
        turn = diag.get("mean_turnover")
        return (
            isinstance(drag, float)
            and isinstance(cost, float)
            and isinstance(turn, float)
            and math.isclose(drag, cost * turn, rel_tol=1e-9)
        )

    out["eval_cost_drag_is_cost_times_turnover"] = _attempt(_cost_drag_formula)

    def _empty_scored_horizon() -> bool:
        times = sorted(set(bars["event_time"].to_list()))[:40]
        rows_h1 = [
            {
                "event_time": t,
                "security_id": sid,
                "horizon_bars": 1,
                "predicted_return": 0.0,
            }
            for t in times
            for sid in ("AAA", "BBB", "CCC")
        ]
        # h2 rows sit in the earliest (train-region) decision times so the
        # horizon is present in the frame but never lands in a test fold.
        rows_h2 = [
            {
                "event_time": t,
                "security_id": sid,
                "horizon_bars": 2,
                "predicted_return": 0.0,
            }
            for t in times[:2]
            for sid in ("AAA", "BBB", "CCC")
        ]
        frame = pl.DataFrame(rows_h1 + rows_h2).with_columns(pl.col("horizon_bars").cast(pl.Int64))
        report2 = evaluate_forecasts(frame, bars, _config(tmp / "h2"))
        return bool(report2["forecast_metrics"]["by_horizon"]["2"] == {"n": 0})

    out["eval_empty_scored_horizon_reports_zero"] = _attempt(_empty_scored_horizon)
    return out


def forecast_core_audit() -> dict[str, Any]:
    """Run every probe; return ``probe name -> bool | str`` in group order."""
    out: dict[str, Any] = {}
    out.update(_config_probes())
    out.update(_schema_probes())
    out.update(_feature_probes())
    out.update(_registry_probes())
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        out.update(_runner_probes(tmp))
        out.update(_eval_probes(tmp))
    return out


def forecast_core_audit_bench() -> dict[str, Any]:
    """Seal the probe map in a ``forecast_core_audit.v1`` receipt."""
    results = forecast_core_audit()
    ok = all(bool(value) for value in results.values())
    out: dict[str, Any] = {
        "kind": "forecast_core_audit",
        "schema": "forecast_core_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": results, "ok": ok},
        "interpretation": (
            "Adversarial audit of the fx-1 forecast core (config/schema/"
            "features/registry/runner/evaluate). Real defects found and "
            "fixed: SignalSection and map_signals admitted non-finite "
            "threshold/cost_bps (nan < 0 is False, so a nan cost surfaced "
            "as a null diagnostic and an inf threshold silently zeroed "
            "every signal) — both now require finite input; "
            "InferenceSection accepted identical output_parquet and "
            "output_meta paths, which let the meta JSON overwrite the "
            "parquet artifact — now refused; FeatureSection accepted "
            "duplicate lookbacks that only failed later inside the "
            "pipeline — now rejected at validation; load_bars surfaced a "
            "raw ColumnNotFoundError for bars without the PIT columns "
            "before the fail-closed contract ran — it now raises "
            "PointInTimeError naming the missing columns. Pinned "
            "legitimate-but-surprising behavior as flag_* probes: "
            "data.start filters decision times but leaves earlier history "
            "visible to features; data.end filters both; quantile ordering "
            "is checked only on rows where every quantile is present "
            "(partial-null rows skip it); hit_rate excludes zero-sign "
            "pairs; the PT statistic needs >= 20 pairs and fails closed on "
            "degenerate margins; rank mapping maps a single name to 0.0; "
            "batch predict sees the whole feature window under a "
            "row-local contract; walk-forward builds features once when "
            "bars are punctual and rebuilds per decision on late releases; "
            "decisions with no feature rows are skipped silently; null and "
            "non-finite signals are skipped in the turnover diagnostic "
            "rather than erroring; turnover is the two-sided sum|dw| "
            "convention; run metadata carries a fresh run_id per run while "
            "forecasts stay deterministic under the synthetic seed; a JSON "
            "checkpoint loads without the unsafe opt-in; mixed sources "
            "label 'mixed'. All probe data is synthetic; no orders."
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
