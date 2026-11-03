"""Fail-closed inputs for fx-1 backtest labels, research traces, and signal weights."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from pydantic import ValidationError

from fx1.forecast.evaluate import _signal_diagnostics, forward_simple_returns
from fx1.forecast.features import OhlcvFeaturePipeline, resample_ohlcv
from fx1.forecast.schema import SchemaError
from fx1.hypotheses import GateVerdict, ResearchTrace, validate_trace_scores
from quant_fund.schemas.errors import PointInTimeError

T0 = datetime(2020, 1, 1, tzinfo=UTC)


def _times(n: int) -> list[datetime]:
    return [T0 + timedelta(days=i) for i in range(n)]


def _closes(values: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": ["AAA"] * len(values),
            "event_time": _times(len(values)),
            "close": values,
        }
    )


def _feature_bars(values: list[float | None]) -> pl.DataFrame:
    times = _times(len(values))
    return _closes(values).with_columns(pl.Series("available_time", times))


def _ohlcv(close: float) -> pl.DataFrame:
    stamp = datetime(2020, 1, 1, 10, tzinfo=UTC)
    return pl.DataFrame(
        {
            "security_id": ["AAA"],
            "event_time": [stamp],
            "available_time": [stamp],
            "open": [close],
            "high": [close],
            "low": [close],
            "close": [close],
            "volume": [1.0],
        }
    )


def _signals(ids: list[str], signals: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [T0] * len(ids),
            "security_id": ids,
            "signal": signals,
        }
    )


def test_forward_labels_keep_a_finite_positive_close_and_a_missing_tail() -> None:
    labels = forward_simple_returns(_closes([100.0, 110.0, 121.0]), 1)
    assert labels["realized_return"][0] == pytest.approx(0.1)
    assert labels["realized_return"][2] is None
    assert labels["target_time"][0] == T0 + timedelta(days=1)


@pytest.mark.parametrize("close", [None, float("nan"), float("inf"), 0.0, -5.0])
def test_forward_labels_reject_unusable_closes(close: float | None) -> None:
    with pytest.raises(PointInTimeError, match="finite and positive"):
        forward_simple_returns(_closes([100.0, close]), 1)


def test_forward_labels_reject_non_numeric_close_and_duplicate_keys() -> None:
    text = _closes([100.0, 110.0]).with_columns(pl.col("close").cast(pl.Utf8))
    with pytest.raises(PointInTimeError, match="numeric"):
        forward_simple_returns(text, 1)
    dup = pl.concat([_closes([100.0]), _closes([101.0])])
    with pytest.raises(PointInTimeError, match="duplicate"):
        forward_simple_returns(dup, 1)
    with pytest.raises(PointInTimeError, match="missing close"):
        forward_simple_returns(_closes([100.0, 110.0]).drop("close"), 1)


def test_feature_bars_reject_null_and_non_positive_closes() -> None:
    pipeline = OhlcvFeaturePipeline([1], 2)
    with pytest.raises(PointInTimeError, match="finite and positive"):
        pipeline.build(_feature_bars([100.0, None, 102.0]))
    with pytest.raises(PointInTimeError, match="finite and positive"):
        pipeline.build(_feature_bars([100.0, -5.0, 102.0]))
    built = pipeline.build(_feature_bars([100.0, 101.0, 102.0, 103.0]))
    assert built.height > 0
    assert "ret_1" in built.columns


def test_resample_rejects_a_non_positive_close() -> None:
    with pytest.raises(PointInTimeError, match="finite and positive"):
        resample_ohlcv(_ohlcv(-1.0), "1d")
    out = resample_ohlcv(_ohlcv(10.0), "1d")
    assert out["close"].to_list() == [10.0]


def test_signal_weights_reject_duplicate_names_and_bad_cost() -> None:
    with pytest.raises(SchemaError, match="duplicate"):
        _signal_diagnostics(_signals(["AAA", "AAA"], [1.0, -1.0]), cost_bps=5.0, stride=1)
    clean = _signals(["AAA", "BBB"], [1.0, None])
    with pytest.raises(ValueError, match="cost_bps"):
        _signal_diagnostics(clean, cost_bps=float("nan"), stride=1)
    with pytest.raises(ValueError, match="cost_bps"):
        _signal_diagnostics(clean, cost_bps=-1.0, stride=1)
    with pytest.raises(ValueError, match="stride"):
        _signal_diagnostics(clean, cost_bps=5.0, stride=0)
    diag = _signal_diagnostics(clean, cost_bps=5.0, stride=1)
    assert diag["orders_submitted"] is False
    assert diag["mean_turnover"] == pytest.approx(1.0)


def test_research_trace_rejects_blank_text_bad_receipt_and_empty_scores() -> None:
    base = {
        "hypothesis": "purged CV reduces overfit",
        "config_diff": "features:\n  horizon_bars: 1",
        "bench_command": "research",
        "scores": {"crps": 0.2},
        "receipt_sha256": "a" * 64,
    }
    with pytest.raises(ValidationError):
        ResearchTrace(**{**base, "hypothesis": "  "})
    with pytest.raises(ValidationError):
        ResearchTrace(**{**base, "config_diff": ""})
    with pytest.raises(ValidationError):
        ResearchTrace(**{**base, "receipt_sha256": "g" * 64})
    with pytest.raises(ValueError, match="at least one"):
        validate_trace_scores({})
    with pytest.raises(ValueError, match="not finite"):
        validate_trace_scores({"crps": True})
    trace = ResearchTrace(**base, verdict=GateVerdict.REJECTED)
    assert trace.admissible
    assert trace.receipt_sha256 == "a" * 64
