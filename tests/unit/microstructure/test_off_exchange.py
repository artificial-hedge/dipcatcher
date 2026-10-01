"""SYNTHETIC print/quote/annotation fixtures; never evidence of hidden intent."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from quant_fund.microstructure.off_exchange import (
    FEATURE_NAMES,
    INDICATOR_DEFINITION,
    PHASES,
    ClassifierConfig,
    ConsolidatedPrint,
    DataProvenance,
    LitQuote,
    OffExchangePhaseClassifier,
    OffExchangeWindow,
    VenueAttribution,
    WindowAnnotation,
    WindowConfig,
    analyze_off_exchange,
    classify_conditions,
    parse_consolidated_print,
)

START = datetime(2025, 1, 1, 12, tzinfo=UTC)
SHA = hashlib.sha256(b"SYNTHETIC fixtures").hexdigest()
PROVENANCE = DataProvenance(
    ("SYNTHETIC_PRINTS", "SYNTHETIC_QUOTES"),
    SHA,
    "SYNTHETIC_GENERATOR_NO_MARKET_ENTITLEMENT",
    False,
    True,
)


def print_(**kwargs: Any) -> ConsolidatedPrint:
    values: dict[str, Any] = {
        "security_id": "SYNTHETIC_A",
        "print_id": "p",
        "source_id": "SYNTHETIC_PRINTS",
        "condition_schema": "UTP_3.0a",
        "message_type": "TM",
        "event_time": START,
        "published_at": START + timedelta(milliseconds=100),
        "price": 100.8,
        "size": 100.0,
        "market_center_id": "D",
        "off_exchange": True,
        "sale_conditions": "@   ",
    }
    values.update(kwargs)
    return ConsolidatedPrint(**values)


def quote(**kwargs: Any) -> LitQuote:
    values: dict[str, Any] = {
        "security_id": "SYNTHETIC_A",
        "quote_id": "q",
        "source_id": "SYNTHETIC_QUOTES",
        "event_time": START - timedelta(milliseconds=500),
        "published_at": START - timedelta(milliseconds=450),
        "bid": 99.0,
        "ask": 101.0,
    }
    values.update(kwargs)
    return LitQuote(**values)


def window(
    prints: list[ConsolidatedPrint] | None = None,
    quotes: list[LitQuote] | None = None,
    *,
    cutoff: datetime = START + timedelta(seconds=1),
    config: WindowConfig | None = None,
) -> OffExchangeWindow:
    return analyze_off_exchange(
        [print_()] if prints is None else prints,
        [quote()] if quotes is None else quotes,
        security_id="SYNTHETIC_A",
        cutoff=cutoff,
        provenance=PROVENANCE,
        config=config,
    )


def annotation(w: OffExchangeWindow, phase_index: int, annotation_id: str) -> WindowAnnotation:
    return WindowAnnotation(
        annotation_id,
        w.receipt_sha256,
        PHASES[phase_index],
        w.start,
        w.cutoff,
        w.cutoff + timedelta(minutes=8),
        "SYNTHETIC_INDEPENDENT_STATE",
        "Labels injected from latent generator state before print generation; synthetic only",
        True,
        True,
    )


def episode(index: int, latent_class: int) -> OffExchangeWindow:
    cutoff = START + timedelta(hours=index)
    prints, quotes = [], []
    for j in range(8):
        event = cutoff - timedelta(minutes=3) + timedelta(seconds=20 * j)
        # An independently supplied generator state determines prints and labels.
        # The production learner has no fallback deriving labels from features.
        side = 1 if latent_class == 0 else -1 if latent_class == 1 else (1 if j % 2 else -1)
        p = print_(
            print_id=f"ep{index}p{j}",
            event_time=event,
            published_at=event + timedelta(milliseconds=100),
            price=100 + 0.7 * side,
            size=100 + (index * 13 + j * 7) % 40,
        )
        q = quote(
            quote_id=f"ep{index}q{j}",
            event_time=event - timedelta(milliseconds=500),
            published_at=event - timedelta(milliseconds=450),
        )
        prints.append(p)
        quotes.append(q)
    return window(prints, quotes, cutoff=cutoff)


@pytest.fixture(scope="module")
def fitted() -> tuple[
    OffExchangePhaseClassifier,
    list[OffExchangeWindow],
    list[WindowAnnotation],
    list[OffExchangeWindow],
    list[WindowAnnotation],
    datetime,
    datetime,
]:
    train = [episode(i, i % 3) for i in range(24)]
    calibration = [episode(i, i % 3) for i in range(24, 42)]
    train_labels = [annotation(w, i % 3, f"train{i}") for i, w in enumerate(train)]
    cal_labels = [annotation(w, i % 3, f"cal{i}") for i, w in enumerate(calibration, start=24)]
    train_cutoff = train[-1].cutoff + timedelta(minutes=10)
    cal_cutoff = calibration[-1].cutoff + timedelta(minutes=10)
    model = OffExchangePhaseClassifier(
        ClassifierConfig(regularization_c=10, platt_c=10, alert_threshold=0.6)
    )
    model.fit(
        train,
        train_labels,
        calibration_windows=calibration,
        calibration_annotations=cal_labels,
        training_cutoff=train_cutoff,
        calibration_cutoff=cal_cutoff,
    )
    return model, train, train_labels, calibration, cal_labels, train_cutoff, cal_cutoff


def test_parser_preserves_source_bytes_and_fractional_size() -> None:
    row = asdict(print_(size=0.5))
    row["event_time"] = START.isoformat()
    row["published_at"] = (START + timedelta(milliseconds=100)).isoformat()
    p = parse_consolidated_print(row, source_id="SYNTHETIC_PRINTS", condition_schema="UTP_3.0a")
    assert p.sale_conditions == "@   " and p.size == 0.5
    with pytest.raises(ValueError, match="finite numeric"):
        parse_consolidated_print(
            {**row, "size": "0.5"}, source_id=p.source_id, condition_schema="UTP_3.0a"
        )
    with pytest.raises(ValueError, match="boolean"):
        parse_consolidated_print(
            {**row, "off_exchange": "false"}, source_id=p.source_id, condition_schema="UTP_3.0a"
        )
    with pytest.raises(ValueError, match="missing print field"):
        parse_consolidated_print({}, source_id=p.source_id, condition_schema="UTP_3.0a")


def test_utp_and_cts_b_have_distinct_meanings() -> None:
    utp = classify_conditions(print_(sale_conditions="@  B"))
    cts = classify_conditions(
        print_(condition_schema="CTS_INPUT_2.7f", message_type="T:L", sale_conditions="   B")
    )
    assert "bunched" in utp.descriptions and "average_price" in cts.descriptions
    assert "special_execution_volume_only" in utp.reasons
    assert "non_current_price_condition_volume_only" in cts.reasons
    assert utp.retained_volume and cts.retained_volume
    assert not utp.permits_quote_sign and not cts.permits_quote_sign
    assert classify_conditions(print_(sale_conditions="@F  ")).permits_quote_sign
    assert classify_conditions(
        print_(condition_schema="CTS_INPUT_2.7f", message_type="T:L", sale_conditions="    ")
    ).permits_quote_sign


@pytest.mark.parametrize("conditions", ["@  ?", "@  F", "@8  ", "N   ", "@  E"])
def test_unknown_mispositioned_reserved_conditions_fail_closed(conditions: str) -> None:
    w = window([print_(sale_conditions=conditions)])
    assert w.off_exchange_volume == 0 and not w.available
    assert not w.audits[0].retained_volume
    assert w.audits[0].quote_sign is None


@pytest.mark.parametrize(
    "conditions,reason",
    [
        ("@ L ", "late_or_out_of_sequence_volume_only"),
        ("@ Z ", "late_or_out_of_sequence_volume_only"),
        ("@ T ", "extended_hours_volume_only"),
        ("@  W", "non_current_price_condition_volume_only"),
        ("@  I", "special_execution_volume_only"),
        ("@  D", "special_execution_volume_only"),
    ],
)
def test_recognized_special_conditions_retain_volume_without_side(
    conditions: str, reason: str
) -> None:
    w = window([print_(sale_conditions=conditions)])
    assert w.off_exchange_volume == 100 and w.available
    assert reason in w.audits[0].reasons
    assert w.off_exchange_signed_volume == 0 and w.positive_quote_proxy_share is None
    assert OffExchangePhaseClassifier().predict(w).phase is None


def test_delayed_prints_and_official_price_messages() -> None:
    p = print_(published_at=START + timedelta(seconds=20))
    w = window([p], cutoff=START + timedelta(seconds=21))
    assert w.off_exchange_volume == 100 and w.off_exchange_signed_volume == 0
    assert "reporting_delay_volume_only" in w.audits[0].reasons
    for conditions in ("@9  ", "@  M", "@  Q"):
        assert not window([print_(sale_conditions=conditions, size=0.0)]).available


def test_cancellation_and_correction_are_causal_and_do_not_double_count() -> None:
    p = print_()
    cancel = print_(
        print_id="cancel",
        message_type="TO",
        reference_print_id="p",
        published_at=START + timedelta(seconds=4),
    )
    before = window([p, cancel], cutoff=START + timedelta(seconds=3))
    baseline = window([p], cutoff=START + timedelta(seconds=3))
    assert before.receipt_sha256 == baseline.receipt_sha256
    assert before.off_exchange_volume == 100
    after = window([p, cancel], cutoff=START + timedelta(seconds=5))
    assert after.off_exchange_volume == 0
    assert any("original_invalidated_by_published_cancel" in a.reasons for a in after.audits)
    corrected = replace(cancel, print_id="correct", message_type="TP", size=200.0)
    after_correction = window([p, corrected], cutoff=START + timedelta(seconds=5))
    assert after_correction.off_exchange_volume == 0
    assert any("correct_message_excluded" in a.reasons for a in after_correction.audits)
    with pytest.raises(ValueError, match="reference_print_id"):
        print_(message_type="TO")


def test_off_exchange_is_not_automatically_ats_or_dark() -> None:
    raw = print_()
    w = window([raw])
    assert w.audits[0].venue_class == "unknown_off_exchange"
    evidence = VenueAttribution(
        "ats",
        "SYNTHETIC_ATS",
        START - timedelta(days=1),
        START + timedelta(days=1),
        START - timedelta(hours=1),
        "SYNTHETIC independently supplied venue attribution",
        SHA,
    )
    assert (
        window([replace(raw, attribution=evidence)]).audits[0].venue_class == "ats_venue_identified"
    )
    assert (
        window([replace(raw, attribution=replace(evidence, venue_id=None))]).audits[0].venue_class
        == "ats_unattributed"
    )
    nonats = replace(evidence, kind="non_ats", venue_id="SYNTHETIC_INTERNALIZER")
    assert (
        window([replace(raw, attribution=nonats)]).audits[0].venue_class
        == "non_ats_venue_identified"
    )
    late = replace(evidence, available_time=START + timedelta(days=1))
    assert window([replace(raw, attribution=late)]).receipt_sha256 == w.receipt_sha256
    stale = replace(evidence, valid_to=START - timedelta(minutes=1))
    assert window([replace(raw, attribution=stale)]).audits[0].venue_class == "unknown_off_exchange"
    with pytest.raises(ValueError, match="mapping mismatch"):
        replace(raw, off_exchange=False)


@pytest.mark.parametrize(
    "quote_kwargs,reason",
    [
        (
            {
                "event_time": START - timedelta(seconds=2),
                "published_at": START - timedelta(seconds=2),
            },
            "stale_lit_quote",
        ),
        (
            {"published_at": START + timedelta(milliseconds=1)},
            "no_contemporaneous_published_lit_quote",
        ),
        ({"bid": 101.0, "ask": 99.0}, "locked_or_crossed_lit_quote"),
        ({"bid": 100.0, "ask": 100.0}, "locked_or_crossed_lit_quote"),
    ],
)
def test_quote_alignment_fail_closed(quote_kwargs: dict[str, Any], reason: str) -> None:
    w = window(quotes=[quote(**quote_kwargs)])
    assert w.off_exchange_volume == 100 and w.off_exchange_signed_volume == 0
    assert reason in w.audits[0].reasons


def test_conflicting_same_clock_quotes_do_not_get_arbitrary_side() -> None:
    quotes = [quote(), quote(quote_id="other", bid=98.0, ask=100.0)]
    w = window(quotes=quotes)
    reversed_w = window(quotes=list(reversed(quotes)))
    assert "ambiguous_contemporaneous_quotes" in w.audits[0].reasons
    assert w.off_exchange_signed_volume == reversed_w.off_exchange_signed_volume == 0


def test_disclosed_quote_proxy_and_lit_measures_do_not_infer_intent() -> None:
    prints = [
        print_(print_id="positive", size=300.0),
        print_(print_id="negative", price=99.2, size=100.0),
        print_(print_id="midpoint", price=100.0, size=10000.0),
        print_(print_id="lit", market_center_id="Q", off_exchange=False, size=200.0),
    ]
    w = window(prints)
    assert w.off_exchange_volume == 10400
    assert w.off_exchange_signed_volume == 200 and w.lit_signed_volume == 200
    assert w.positive_quote_proxy_share == 0.75
    assert w.indicator_definition == INDICATOR_DEFINITION
    assert w.buyer_intent == "unknown" and w.market_evidence is False
    assert len(w.features) == len(FEATURE_NAMES)
    signal = OffExchangePhaseClassifier().predict(w)
    assert signal.status == "unavailable" and signal.probabilities is None
    assert signal.buyer_intent == "unknown" and signal.phase is None
    assert OffExchangePhaseClassifier().alert(w) is None


def test_future_suffix_and_unpublished_data_do_not_change_receipts() -> None:
    w = window()
    p = print_(
        print_id="future",
        event_time=START + timedelta(seconds=2),
        published_at=START + timedelta(seconds=3),
        size=100000.0,
    )
    q = quote(
        quote_id="future",
        event_time=START + timedelta(seconds=2),
        published_at=START + timedelta(seconds=3),
        bid=1.0,
        ask=2.0,
    )
    extended = window([print_(), p], [quote(), q])
    assert extended.receipt_sha256 == w.receipt_sha256
    assert extended.data_sha256 == w.data_sha256
    np.testing.assert_array_equal(extended.features, w.features)
    unpublished = replace(p, event_time=START - timedelta(minutes=1))
    assert window([print_(), unpublished]).receipt_sha256 == w.receipt_sha256


def test_hashes_data_rights_source_and_synthetic_identity() -> None:
    w = window()
    assert w.provenance.synthetic and len(w.receipt_sha256) == 64
    assert window([print_(size=101.0)]).data_sha256 != w.data_sha256
    assert window(config=WindowConfig(large_print_size=1.0)).config_sha256 != w.config_sha256
    modified = replace(w, features=(0.0,) * len(FEATURE_NAMES))
    assert modified.receipt_sha256 != w.receipt_sha256
    with pytest.raises(ValueError, match="entitlement"):
        analyze_off_exchange(
            [print_()],
            [quote()],
            security_id="SYNTHETIC_A",
            cutoff=START,
            provenance=replace(PROVENANCE, synthetic=False),
        )
    with pytest.raises(ValueError, match="source missing"):
        analyze_off_exchange(
            [print_(source_id="UNDECLARED")],
            [],
            security_id="SYNTHETIC_A",
            cutoff=START,
            provenance=PROVENANCE,
        )
    with pytest.raises(ValueError, match="duplicate visible"):
        window([print_(), print_()])
    with pytest.raises(ValueError, match="timezone-aware"):
        print_(event_time=START.replace(tzinfo=None))


def test_dst_fold_future_print_is_not_visible() -> None:
    zone = ZoneInfo("America/New_York")
    cutoff = datetime(2025, 11, 2, 1, 30, tzinfo=zone, fold=0)
    event = datetime(2025, 11, 2, 1, 10, tzinfo=zone, fold=1)
    w = window([print_(event_time=event, published_at=event)], [], cutoff=cutoff)
    assert w.audits == () and not w.available


def test_missing_labels_yield_unavailable_classification() -> None:
    w = window()
    model = OffExchangePhaseClassifier().fit(
        [w],
        None,
        calibration_windows=[],
        calibration_annotations=None,
        training_cutoff=START,
        calibration_cutoff=START + timedelta(days=1),
    )
    assert model.model_sha256 is None and model.predict(w).status == "unavailable"
    with pytest.raises(ValueError, match="cannot fabricate"):
        model.evaluate([w], [annotation(w, 0, "a")], asof=START + timedelta(days=1))
    with pytest.raises(ValueError, match="independent methodology"):
        replace(annotation(w, 0, "a"), independent_of_print_features=False)


def test_actual_calibrated_learning_proper_holdout_scores_and_local_alerts(
    fitted: tuple[Any, ...],
) -> None:
    model = fitted[0]
    test = [episode(i, i % 3) for i in range(42, 57)]
    labels = [annotation(w, i % 3, f"test{i}") for i, w in enumerate(test, start=42)]
    signals = [model.predict(w) for w in test]
    assert sum(s.phase == a.phase for s, a in zip(signals, labels, strict=True)) >= 13
    for s in signals:
        assert s.probabilities is not None
        assert sum(s.probabilities) == pytest.approx(1.0)
        assert s.synthetic and s.buyer_intent == "unknown" and not s.market_evidence
    scores = model.evaluate(test, labels, asof=test[-1].cutoff + timedelta(minutes=10))
    assert scores["multiclass_brier"] < 0.35 and scores["log_score"] < 0.7
    assert 0 <= scores["ece"] <= 1 and all(math.isfinite(v) for v in scores.values())
    alert = model.alert(test[0])
    assert alert is not None
    assert alert.local_only and alert.research_only and alert.synthetic
    assert alert.buyer_intent == "unknown" and not alert.market_evidence
    assert alert.model_sha256 == model.model_sha256 and len(alert.alert_sha256) == 64
    assert model.alert(test[2]) is None  # independently supplied no_signal target.
    old_hash = model.model_sha256
    assert model.metadata()["data_label"] == "SYNTHETIC"
    assert model.metadata()["market_evidence"] is False
    model.predict(test[-1])
    assert old_hash == model.model_sha256 and not model._mean.flags.writeable
    with pytest.raises(AttributeError):
        model.config = ClassifierConfig()  # type: ignore[misc]


def test_label_alignment_publication_chronology_and_config_fail_closed(
    fitted: tuple[Any, ...],
) -> None:
    _, train, labels, calibration, cal_labels, train_cutoff, cal_cutoff = fitted

    def fit(
        new_train: Any = train,
        new_labels: Any = labels,
        new_calibration: Any = calibration,
        new_cal_labels: Any = cal_labels,
        new_train_cutoff: datetime = train_cutoff,
    ) -> None:
        OffExchangePhaseClassifier().fit(
            new_train,
            new_labels,
            calibration_windows=new_calibration,
            calibration_annotations=new_cal_labels,
            training_cutoff=new_train_cutoff,
            calibration_cutoff=cal_cutoff,
        )

    with pytest.raises(ValueError, match="alignment mismatch"):
        fit(new_labels=[replace(labels[0], window_sha256="a" * 64), *labels[1:]])
    with pytest.raises(ValueError, match="unpublished"):
        fit(
            new_labels=[
                replace(labels[0], available_time=train_cutoff + timedelta(seconds=1)),
                *labels[1:],
            ]
        )
    with pytest.raises(ValueError, match="synthetic identity"):
        fit(new_labels=[replace(labels[0], synthetic=False), *labels[1:]])
    with pytest.raises(ValueError, match="strictly after training"):
        fit(new_train_cutoff=calibration[0].cutoff)
    with pytest.raises(ValueError, match="disjoint"):
        fit(
            new_cal_labels=[
                replace(cal_labels[0], annotation_id=labels[0].annotation_id),
                *cal_labels[1:],
            ]
        )
    altered = replace(train[1], start=train[0].cutoff - timedelta(seconds=1))
    altered_label = replace(
        labels[1], window_sha256=altered.receipt_sha256, observed_start=altered.start
    )
    with pytest.raises(ValueError, match="cannot overlap"):
        fit(
            new_train=[train[0], altered, *train[2:]],
            new_labels=[labels[0], altered_label, *labels[2:]],
        )
    model = fitted[0]
    with pytest.raises(ValueError, match="after calibration"):
        model.predict(calibration[0])
    with pytest.raises(ValueError, match="configuration mismatch"):
        model.predict(replace(episode(42, 0), config_sha256="b" * 64))
    with pytest.raises(RuntimeError, match="frozen"):
        model.fit(
            train,
            labels,
            calibration_windows=calibration,
            calibration_annotations=cal_labels,
            training_cutoff=train_cutoff,
            calibration_cutoff=cal_cutoff,
        )


def test_single_class_annotations_cannot_claim_multiclass_calibration(
    fitted: tuple[Any, ...],
) -> None:
    _, train, labels, calibration, cal_labels, train_cutoff, cal_cutoff = fitted
    with pytest.raises(ValueError, match="at least two independent labels per class"):
        OffExchangePhaseClassifier().fit(
            train,
            [replace(a, phase="accumulation") for a in labels],
            calibration_windows=calibration,
            calibration_annotations=cal_labels,
            training_cutoff=train_cutoff,
            calibration_cutoff=cal_cutoff,
        )
