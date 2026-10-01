"""Options flow tests use synthetic tape and invented annotation labels only."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from quant_fund.models.options import bs_greeks
from quant_fund.models.options_flow import (
    IntentAnnotation,
    OptionsFlowConfig,
    OptionsFlowModel,
    OptionTrade,
    alert_unusual,
    ingest_option_tape,
    trade_features,
)


def _fixture(n: int = 220) -> tuple[list[OptionTrade], list[IntentAnnotation]]:
    base = datetime(2020, 1, 1, tzinfo=UTC)
    rng = np.random.default_rng(16)
    trades, labels = [], []
    for i in range(n):
        event = base + timedelta(minutes=i)
        high = i % 2 == 0
        size = (900 if high else 90) + int(rng.integers(1, 30))
        trade = OptionTrade(
            trade_id=f"t{i}",
            contract_id="ABC20210101C100",
            underlying="ABC",
            strike=100.0,
            option_type="call",
            expiry=datetime(2021, 1, 1, tzinfo=UTC),
            event_time=event,
            available_time=event + timedelta(milliseconds=10),
            price=3.05 if high else 2.95,
            size=size,
            implied_vol=0.2,
            underlying_price=100.0 + float(rng.normal(scale=0.1)),
            underlying_event_time=event - timedelta(seconds=2),
            underlying_available_time=event - timedelta(seconds=1),
            bid=2.9,
            ask=3.1,
            quote_event_time=event - timedelta(seconds=1),
            quote_available_time=event,
            data_source="synthetic_options_correctness",
            synthetic=True,
        )
        intent = "informed" if high else "hedge"
        # Invented labels deliberately depend on size/price; they demonstrate
        # learning a supplied annotation convention, never true trade intent.
        labels.append(
            IntentAnnotation(
                trade.trade_id,
                intent,
                event + timedelta(seconds=20),
                "invented_correctness_labels",
                hashlib.sha256(f"synthetic-{i}-{intent}".encode()).hexdigest(),
                True,
            )
        )
        trades.append(trade)
    return trades, labels


def _config(**changes: Any) -> OptionsFlowConfig:
    return replace(OptionsFlowConfig(n_estimators=12), **changes)


def _fit(
    trades: list[OptionTrade], labels: list[IntentAnnotation], **changes: Any
) -> OptionsFlowModel:
    return OptionsFlowModel(_config(**changes)).fit(
        trades,
        labels,
        train_cutoff=trades[119].event_time + timedelta(seconds=40),
        cutoff=trades[159].event_time + timedelta(seconds=40),
    )


def test_strict_parser_roundtrip_and_clock_preservation() -> None:
    trade = _fixture(1)[0][0]
    payload = asdict(trade)
    parsed = ingest_option_tape(json.dumps(payload, default=lambda x: x.isoformat()))
    assert parsed == trade and parsed.source_sha256 == trade.source_sha256
    assert parsed.event_time != parsed.available_time
    assert parsed.quote_available_time.tzinfo is UTC
    offset = dict(payload)
    offset["event_time"] = trade.event_time.astimezone(UTC).isoformat().replace("+00:00", "Z")
    assert ingest_option_tape(offset).event_time == trade.event_time


@pytest.mark.parametrize(
    "change",
    [
        {"size": True},
        {"size": 0},
        {"size": 10_000_001},
        {"size": "2"},
        {"price": math.nan},
        {"implied_vol": 0},
        {"ask": 1.0},
        {"option_type": "CALL"},
        {"synthetic": "false"},
        {"multi_leg": "false"},
        {"risk_free_rate": "bad"},
        {"risk_free_rate": 2},
        {"data_source": ""},
        {"contract_id": " padded "},
    ],
)
def test_invalid_tape_values_fail_closed(change: dict[str, Any]) -> None:
    payload = asdict(_fixture(1)[0][0])
    payload.update(change)
    with pytest.raises(ValueError):
        ingest_option_tape(payload)


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "unknown",
        "duplicate_json",
        "naive",
        "future_quote",
        "underlying_not_available",
        "expired",
        "availability_before_event",
        "array",
    ],
)
def test_parser_schema_and_temporal_errors(case: str) -> None:
    trade = _fixture(1)[0][0]
    payload = asdict(trade)
    if case == "missing":
        del payload["contract_id"]
    elif case == "unknown":
        payload["hidden_label"] = "informed"
    elif case == "duplicate_json":
        with pytest.raises(ValueError, match="duplicate"):
            ingest_option_tape('{"size": 2, "size": 3}')
        return
    elif case == "naive":
        payload["event_time"] = trade.event_time.replace(tzinfo=None)
    elif case == "future_quote":
        payload["quote_event_time"] = trade.event_time + timedelta(seconds=1)
    elif case == "underlying_not_available":
        payload["underlying_available_time"] = trade.available_time + timedelta(seconds=1)
    elif case == "expired":
        payload["expiry"] = trade.event_time
    elif case == "availability_before_event":
        payload["available_time"] = trade.event_time - timedelta(seconds=1)
    elif case == "array":
        with pytest.raises(ValueError, match="object"):
            ingest_option_tape("[]")
        return
    with pytest.raises(ValueError):
        ingest_option_tape(payload)


def test_greeks_and_trailing_only_features_do_not_consume_current_or_future_sizes() -> None:
    trades, _ = _fixture()
    query = trades[15]
    features = trade_features(query, trades, _config())
    expected = bs_greeks(
        query.underlying_price,
        query.strike,
        (query.expiry - query.event_time).total_seconds() / (365.25 * 86_400),
        query.implied_vol,
    )
    assert features.values[2] == pytest.approx(expected["delta"])
    assert features.values[3] == pytest.approx(expected["gamma"])
    sizes = np.asarray([t.size for t in trades[:15]])
    assert features.volume_zscore == pytest.approx((query.size - sizes.mean()) / sizes.std())
    future = trades[:16] + [replace(t, size=9_999_999) for t in trades[16:]]
    assert trade_features(query, future, _config()) == features
    assert trade_features(query, trades[:15], _config()) == features
    assert features.block_flag is False and features.n_history == 15


def test_constant_trailing_size_reports_undefined_zscore_with_indicator() -> None:
    trades, _ = _fixture(10)
    history = [replace(t, size=100) for t in trades[:8]]
    features = trade_features(trades[8], history, _config())
    assert features.volume_zscore is None
    assert features.values[9:11] == (0.0, 0.0)


def test_zero_bid_and_put_greeks_are_valid_quote_inputs() -> None:
    trades, _ = _fixture(12)
    put_tape = [replace(t, option_type="put", contract_id="ABC20210101P100", bid=0) for t in trades]
    features = trade_features(put_tape[-1], put_tape[:-1], _config())
    assert -1 <= features.values[2] <= 0
    assert features.values[3] > 0


def test_missing_labels_or_training_does_not_fabricate_confidence() -> None:
    trades, _ = _fixture()
    model = OptionsFlowModel(_config())
    score = model.score_trade(trades[20])
    assert score.status == "untrained" and score.probability_informed_annotation is None
    with pytest.raises(ValueError, match="annotations are required"):
        model.fit(trades, [], train_cutoff=trades[119].event_time, cutoff=trades[159].event_time)
    assert alert_unusual([score], threshold=0.5) == ()


def test_actual_ensemble_calibration_and_independent_holdout_scores() -> None:
    trades, labels = _fixture()
    model = _fit(trades, labels)
    info = model.fit_info
    assert info is not None and info.n_training == 115 and info.n_calibration == 40
    assert len(info.model_sha256) == 64 and len(info.training_data_sha256) == 64
    assert info.synthetic and math.isfinite(info.calibration_brier)
    assert math.isfinite(info.calibration_log_loss) and 0 <= info.calibration_ece <= 1
    high, low = model.score_trade(trades[180], trades), model.score_trade(trades[181], trades)
    assert (
        high.probability_informed_annotation is not None
        and low.probability_informed_annotation is not None
    )
    assert high.probability_informed_annotation > low.probability_informed_annotation + 0.4
    assert not high.actual_intent_established and high.in_sample is False
    assert "supplied" in high.label_semantics and "not observed intent" in high.label_semantics
    report = model.evaluate(trades[160:], labels[160:], label_cutoff=labels[-1].available_time)
    assert report["n_scored"] == 60 and report["brier_score"] < 0.15
    assert report["log_loss"] < 0.6 and 0 <= report["ece"] <= 1
    assert report["synthetic"] and not report["sota_established"]
    assert report["actual_intent_established"] is False
    alerts = alert_unusual([high, low], threshold=0.7)
    assert len(alerts) == 1 and alerts[0]["trade_id"] == high.trade_id
    assert not alerts[0]["actual_intent_established"]
    assert alerts[0]["synthetic"] and alerts[0]["training_synthetic"]
    assert alerts[0]["annotation_sources"] == ("invented_correctness_labels",)


def test_future_suffix_cannot_change_scaling_models_or_calibration() -> None:
    trades, labels = _fixture()
    first = _fit(trades, labels)
    changed = trades[:160] + [replace(t, size=9_999_999, implied_vol=3) for t in trades[160:]]
    changed_labels = labels[:160] + [replace(a, intent="hedge") for a in labels[160:]]
    second = _fit(changed, changed_labels)
    assert first.fit_info == second.fit_info
    assert (
        first.score_trade(trades[180], trades).probability_informed_annotation
        == second.score_trade(trades[180], trades).probability_informed_annotation
    )


def test_label_availability_filters_both_training_and_calibration() -> None:
    trades, labels = _fixture()
    baseline = _fit(trades, labels)
    delayed = labels.copy()
    delayed[20] = replace(labels[20], available_time=trades[180].available_time)
    delayed[140] = replace(labels[140], available_time=trades[180].available_time)
    changed = _fit(trades, delayed)
    assert baseline.fit_info is not None and changed.fit_info is not None
    assert changed.fit_info.n_training == baseline.fit_info.n_training - 1
    assert changed.fit_info.n_calibration == baseline.fit_info.n_calibration - 1
    assert changed.fit_info.model_sha256 != baseline.fit_info.model_sha256


def test_calibration_annotations_do_not_change_base_training_identity() -> None:
    trades, labels = _fixture()
    first = _fit(trades, labels)
    flipped = (
        labels[:120]
        + [
            replace(a, intent="hedge" if a.intent == "informed" else "informed")
            for a in labels[120:160]
        ]
        + labels[160:]
    )
    second = _fit(trades, flipped)
    assert first.fit_info is not None and second.fit_info is not None
    assert first.fit_info.training_data_sha256 == second.fit_info.training_data_sha256
    assert first.fit_info.calibration_data_sha256 != second.fit_info.calibration_data_sha256
    assert first.fit_info.model_sha256 != second.fit_info.model_sha256
    np.testing.assert_array_equal(first._linear.coef_, second._linear.coef_)


def test_unsupported_multileg_stale_quotes_and_contract_conflicts() -> None:
    trades, labels = _fixture()
    model = _fit(trades, labels)
    for query in (
        replace(trades[180], multi_leg=True),
        replace(
            trades[180],
            quote_event_time=trades[180].event_time - timedelta(seconds=100),
            quote_available_time=trades[180].event_time - timedelta(seconds=90),
        ),
    ):
        score = model.score_trade(query, trades)
        assert score.probability_informed_annotation is None
        assert "unsupported" in score.status or "stale" in score.status
    bad = trades[:20] + [replace(trades[20], strike=101)]
    with pytest.raises(ValueError, match="inconsistent option contracts"):
        trade_features(bad[-1], bad[:-1], _config())


def test_missing_class_bad_annotations_refit_and_config_fail_closed() -> None:
    trades, labels = _fixture()
    model = _fit(trades, labels)
    with pytest.raises(ValueError, match="both classes"):
        _fit(trades, [replace(a, intent="hedge") for a in labels])
    with pytest.raises(ValueError, match="annotations are required"):
        model.fit(trades, [], train_cutoff=trades[119].event_time, cutoff=trades[159].event_time)
    assert model.score_trade(trades[180], trades).probability_informed_annotation is None
    with pytest.raises(ValueError, match="uniquely"):
        _fit(trades, labels + [labels[0]])
    with pytest.raises(ValueError, match="before the trade"):
        _fit(
            trades,
            [replace(labels[0], available_time=trades[0].event_time - timedelta(seconds=1))]
            + labels[1:],
        )
    trained = _fit(trades, labels)
    trained.config = _config(lookback=30)
    with pytest.raises(RuntimeError, match="retraining"):
        trained.score_trade(trades[180], trades)


def test_unavailable_score_cannot_be_changed_into_fabricated_confidence() -> None:
    trades, _ = _fixture()
    score = OptionsFlowModel().score_trade(trades[10])
    with pytest.raises(ValueError, match="cannot carry class confidence"):
        replace(score, probability_informed_annotation=0.9)
    with pytest.raises(ValueError, match="requires model"):
        replace(score, status="scored", probability_informed_annotation=0.9)


def test_holdout_cannot_reuse_fit_or_unavailable_labels() -> None:
    trades, labels = _fixture()
    model = _fit(trades, labels)
    with pytest.raises(ValueError, match="follow the model fit cutoff"):
        model.evaluate(trades[:60], labels[:60], label_cutoff=labels[-1].available_time)
    unavailable = [
        replace(a, available_time=labels[-1].available_time + timedelta(days=2))
        for a in labels[160:]
    ]
    with pytest.raises(ValueError, match="ten labeled"):
        model.evaluate(trades[160:], unavailable, label_cutoff=labels[-1].available_time)


@pytest.mark.parametrize("threshold", [0, 1, math.nan, True, -0.2])
def test_alert_threshold_validation(threshold: float) -> None:
    with pytest.raises(ValueError):
        alert_unusual([], threshold=threshold)


@pytest.mark.parametrize(
    "change",
    [
        {"lookback": 1},
        {"min_history": 51},
        {"block_size": True},
        {"n_estimators": 257},
        {"tree_depth": 5},
        {"max_quote_age_seconds": math.inf},
    ],
)
def test_feature_and_learning_resource_bounds(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        OptionsFlowConfig(**change)
