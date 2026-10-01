"""SYNTHETIC release/parser/regression correctness; no empirical evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

from quant_fund.models.economic_events import (
    CalendarEvent,
    ConsensusForecast,
    EconomicEventModel,
    EventConfig,
    EventObservation,
    EventOutcome,
    ReleaseVintage,
    TimedPrice,
    _hash,
    extract_release,
    local_release_time,
    measure_reaction,
)

START = datetime(2020, 1, 1, 8, 30, tzinfo=UTC)
SHA = hashlib.sha256(b"SYNTHETIC independently supplied clocks and labels").hexdigest()
CONFIG = EventConfig("SYNTHETIC_SPY", kinds=("cpi_yoy",), ridge_alpha=0.1)


def row(index: int, *, surprise: float | None = None, synthetic: bool = True) -> EventOutcome:
    rng = np.random.default_rng(index + 123)
    release = START + timedelta(days=index)
    surprise = float(rng.normal(0, 0.002)) if surprise is None else surprise
    calendar = CalendarEvent(
        str(index),
        "CPI_TEST_YOY",
        "cpi_yoy",
        f"period_{index}",
        release,
        release - timedelta(days=10),
        release - timedelta(days=9),
        SHA,
        synthetic,
    )
    vintage = ReleaseVintage(
        str(index),
        f"v{index}",
        0,
        0.03 + surprise,
        "fraction",
        release,
        release + timedelta(seconds=1),
        SHA,
        SHA,
        synthetic,
    )
    consensus = ConsensusForecast(
        str(index),
        f"c{index}",
        0.03,
        "fraction",
        release - timedelta(hours=2),
        release - timedelta(hours=1),
        SHA,
        synthetic,
    )
    prices = tuple(
        TimedPrice(
            f"p{index}_{j}",
            release - timedelta(minutes=10 - j),
            release - timedelta(minutes=9 - j),
            float(100 * math.exp(0.0001 * j + 0.00003 * math.sin(index + j))),
            SHA,
            synthetic,
        )
        for j in range(4)
    )
    observation = EventObservation(
        str(index),
        "SYNTHETIC_SPY",
        calendar,
        (vintage,),
        (consensus,),
        prices,
        release + timedelta(seconds=2),
    )
    end = observation.decision_at + timedelta(hours=1)
    target_return = 2 * surprise + float(rng.normal(0, 0.0003))
    # Coherent supplied SYNTHETIC N=12 interval labels, including the mean move.
    realized_variance = target_return**2 / 12 + math.exp(-12 + 200 * surprise)
    return EventOutcome(
        observation,
        target_return,
        realized_variance,
        end,
        end + timedelta(seconds=1),
        SHA,
        synthetic,
    )


@pytest.fixture
def datasets():
    return (
        tuple(row(i) for i in range(32)),
        tuple(row(i) for i in range(40, 56)),
        tuple(row(i) for i in range(70, 86)),
    )


@pytest.fixture
def fitted(datasets):
    train, calibration, _ = datasets
    return EconomicEventModel.fit(
        train,
        calibration,
        config=CONFIG,
        train_asof=START + timedelta(days=35),
        calibration_asof=START + timedelta(days=60),
    )


@pytest.mark.parametrize(
    ("kind", "text", "expected"),
    [
        ("cpi_yoy", "US CPI YoY; Actual: 3.4%; Consensus: 3.1%", 0.034),
        (
            "cpi_yoy",
            "Over the last 12 months, the all items index increased 3.4 percent before seasonal adjustment.",
            0.034,
        ),
        (
            "cpi_mom",
            "The Consumer Price Index for All Urban Consumers increased 0.4 percent on a seasonally adjusted basis.",
            0.004,
        ),
        ("payroll_change", "Total nonfarm payroll employment rose by 142,000 jobs.", 142000),
        ("payroll_change", "Nonfarm payroll Actual: -120 thousand", -120000),
        (
            "gdp_annualized",
            "Real gross domestic product decreased at an annual rate of 1.2 percent.",
            -0.012,
        ),
        ("policy_rate", "Federal funds policy rate; Actual: 525 bps", 0.0525),
    ],
)
def test_parser_converts_explicit_units_and_preserves_separate_text_hash(kind, text, expected):
    event = replace(row(0).observation.calendar, kind=kind)
    result = extract_release(
        text,
        event,
        vintage_id="parsed",
        revision=0,
        released_at=START,
        available_at=START + timedelta(seconds=1),
        source_sha256=SHA,
        synthetic=True,
    )
    assert result.value == pytest.approx(expected)
    assert result.text_sha256 == hashlib.sha256(text.encode()).hexdigest()
    assert result.source_sha256 == SHA
    assert result.synthetic


@pytest.mark.parametrize(
    "text",
    [
        "CPI YoY Actual: 3.2%; Actual: 3.3%",
        "CPI YoY Actual: 3.2 jobs",
        "CPI YoY Actual: 3.2",
        "CPI YoY might have risen quite a bit",
        "Consumer Price Index includes both annual and monthly rates",
        "Nonfarm payroll Actual: 100000 jobs",
        "CPI YoY Consensus: 3.2%",
    ],
)
def test_uncertain_mismatched_or_postrelease_consensus_text_is_not_actual(text):
    with pytest.raises(ValueError):
        extract_release(
            text,
            row(0).observation.calendar,
            vintage_id="bad",
            revision=0,
            released_at=START,
            available_at=START,
            source_sha256=SHA,
            synthetic=True,
        )


def test_parser_bounded_no_duplicate_or_unknown_event_alias():
    with pytest.raises(ValueError, match="bounded"):
        extract_release(
            "CPI YoY Actual: 2.1%" + " " * 16384,
            row(0).observation.calendar,
            vintage_id="bad",
            revision=0,
            released_at=START,
            available_at=START,
            source_sha256=SHA,
            synthetic=True,
        )
    with pytest.raises(ValueError, match="kind"):
        replace(row(0).observation.calendar, kind="arbitrary")


@pytest.mark.parametrize(
    "kind,text",
    [
        ("payroll_change", "Nonfarm payroll Actual: 1,23 jobs"),
        ("payroll_change", "Nonfarm payroll Actual: 150"),
        ("payroll_change", "Nonfarm payroll Actual: 1.2 persons"),
        ("payroll_change", "Nonfarm payroll employment rose by 142,00 jobs."),
        ("cpi_mom", "Consumer Price Index YoY Actual: 3.2%"),
        ("gdp_annualized", "GDP annualized and CPI YoY; Actual: 3.2%"),
    ],
)
def test_parser_rejects_truncated_numbers_ambiguous_periods_and_multiple_events(kind, text):
    event = replace(row(0).observation.calendar, kind=kind)
    with pytest.raises(ValueError):
        extract_release(
            text,
            event,
            vintage_id="bad",
            revision=0,
            released_at=START,
            available_at=START,
            source_sha256=SHA,
            synthetic=True,
        )


def test_calendar_timezone_dst_and_utc_ordering():
    winter = local_release_time(datetime(2026, 1, 13, 8, 30), "America/New_York")
    summer = local_release_time(datetime(2026, 7, 14, 8, 30), "America/New_York")
    assert winter.hour == 13 and summer.hour == 12
    with pytest.raises(ValueError, match="nonexistent"):
        local_release_time(datetime(2026, 3, 8, 2, 30), "America/New_York")
    ambiguous = datetime(2026, 11, 1, 1, 30)
    with pytest.raises(ValueError, match="ambiguous"):
        local_release_time(ambiguous, "America/New_York")
    assert local_release_time(ambiguous, "America/New_York", fold=1) - local_release_time(
        ambiguous, "America/New_York", fold=0
    ) == timedelta(hours=1)
    with pytest.raises(ValueError, match="timezone"):
        local_release_time(datetime(2026, 1, 1, 1), "Nonexistent/Zone")


@pytest.mark.parametrize(
    "fault",
    [
        "consensus_late",
        "consensus_equal",
        "unit",
        "naive",
        "release_receipt",
        "calendar_late",
        "duplicate",
        "horizon",
        "future_label",
    ],
)
def test_publication_consensus_units_calendar_horizon_fail_closed(datasets, fault):
    train, cal, _ = datasets
    source = train[0]
    obs = source.observation
    if fault in ("consensus_late", "consensus_equal"):
        clock = obs.releases[0].released_at + timedelta(seconds=fault == "consensus_late")
        obs = replace(obs, consensus=(replace(obs.consensus[0], available_at=clock),))
    elif fault == "unit":
        with pytest.raises(ValueError, match="units"):
            replace(obs, consensus=(replace(obs.consensus[0], unit="persons"),))
        return
    elif fault == "naive":
        with pytest.raises(ValueError, match="timezone"):
            replace(obs, decision_at=obs.decision_at.replace(tzinfo=None))
        return
    elif fault == "release_receipt":
        obs = replace(
            obs,
            releases=(
                replace(obs.releases[0], available_at=obs.decision_at + timedelta(seconds=1)),
            ),
        )
    elif fault == "calendar_late":
        with pytest.raises(ValueError, match="calendar"):
            replace(obs.calendar, available_at=obs.decision_at + timedelta(seconds=1))
        return
    elif fault == "duplicate":
        train = (train[0], train[0], *train[2:])
    elif fault == "horizon":
        source = replace(source, target_end=source.target_end - timedelta(minutes=1))
    elif fault == "future_label":
        source = replace(source, available_at=START + timedelta(days=36))
    if fault not in ("horizon", "future_label", "duplicate"):
        source = replace(source, observation=obs)
    if fault != "duplicate":
        train = (source, *train[1:])
    with pytest.raises(ValueError):
        EconomicEventModel.fit(
            train,
            cal,
            config=CONFIG,
            train_asof=START + timedelta(days=35),
            calibration_asof=START + timedelta(days=60),
        )


def test_revision_and_future_price_suffix_do_not_rewrite_past_features(fitted, datasets):
    observed = datasets[2][0].observation
    before = fitted.predict(observed)
    future_revision = replace(
        observed.releases[0],
        vintage_id="revised",
        revision=1,
        value=0.08,
        released_at=observed.decision_at + timedelta(minutes=2),
        available_at=observed.decision_at + timedelta(minutes=3),
    )
    future_price = TimedPrice(
        "futureprice",
        observed.decision_at + timedelta(minutes=5),
        observed.decision_at + timedelta(minutes=6),
        10000,
        SHA,
        True,
    )
    future_consensus = replace(
        observed.consensus[0],
        forecast_id="postrelease",
        value=0.099,
        published_at=observed.decision_at,
        available_at=observed.decision_at + timedelta(minutes=3),
    )
    suffixed = replace(
        observed,
        releases=(*observed.releases, future_revision),
        consensus=(*observed.consensus, future_consensus),
        past_prices=(*observed.past_prices, future_price),
    )
    assert fitted.predict(suffixed) == before
    later = replace(suffixed, decision_at=future_revision.available_at + timedelta(seconds=1))
    later_prediction = fitted.predict(later)
    assert later_prediction.input_sha256 != before.input_sha256
    assert later_prediction.mean_log_return != before.mean_log_return


def test_train_scaling_numeric_fit_matches_independent_linear_algebra(fitted):
    body = fitted._body
    x = np.array([r["snapshot"]["features"] for r in body["training"]])
    y = np.array([r["log_return"] for r in body["training"]])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale <= 1e-12] = 1
    z = np.c_[np.ones(len(x)), (x - mean) / scale]
    penalty = np.eye(z.shape[1]) * CONFIG.ridge_alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    np.testing.assert_allclose(body["learned"]["mean"], mean, rtol=0, atol=0)
    np.testing.assert_allclose(body["learned"]["scale"], scale, rtol=0, atol=0)
    np.testing.assert_allclose(
        body["learned"]["arms"]["surprise"]["beta"], beta, rtol=1e-12, atol=1e-15
    )
    assert np.linalg.norm(beta[1:]) > 0
    # Later calibration features never enter the training scaler.
    assert not np.array_equal(
        mean, np.array([r["snapshot"]["features"] for r in body["calibration"]]).mean(axis=0)
    )


def test_holdout_scores_match_independent_formulas_and_r2_is_diagnostic(fitted, datasets):
    holdout = datasets[2]
    result = fitted.evaluate(holdout, asof=START + timedelta(days=90))
    y = np.array([r.log_return for r in holdout])
    variance = np.array([r.realized_variance for r in holdout])
    for outcome in result["outcomes"]:
        predictions = [fitted.predict(r.observation, method=outcome["method"]) for r in holdout]
        mu = np.array([p.mean_log_return for p in predictions])
        sigma = np.array([p.sigma_log_return for p in predictions])
        v = np.array([p.realized_variance_forecast for p in predictions])
        standardized = (y - mu) / sigma
        pdf = np.exp(-(standardized**2) / 2) / math.sqrt(2 * math.pi)
        crps = sigma * (
            standardized * (2 * ndtr(standardized) - 1) + 2 * pdf - 1 / math.sqrt(math.pi)
        )
        assert outcome["gaussian_crps"] == pytest.approx(crps.mean(), rel=1e-12)
        assert outcome["gaussian_negative_log_score"] == pytest.approx(
            np.mean(np.log(sigma) + math.log(2 * math.pi) / 2 + standardized**2 / 2), rel=1e-12
        )
        assert outcome["variance_qlike_unnormalized"] == pytest.approx(
            np.mean(np.log(v) + variance / v), rel=1e-12
        )
        assert outcome["return_r2_diagnostic"] == pytest.approx(
            1 - np.sum((y - mu) ** 2) / np.sum((y - y.mean()) ** 2), rel=1e-12
        )
    assert result["synthetic"] and not result["market_evidence"]
    assert result["r2_is_diagnostic"] and not result["causal_effect_identified"]
    assert len(result["outcomes"]) == 2


def test_zero_realized_variance_retained_and_undefined_r2_is_not_fabricated(fitted, datasets):
    holdout = tuple(replace(r, log_return=0.0, realized_variance=0.0) for r in datasets[2])
    result = fitted.evaluate(holdout, asof=START + timedelta(days=90))
    for outcome in result["outcomes"]:
        assert outcome["return_r2_diagnostic"] is None
        assert outcome["variance_r2_diagnostic"] is None
        assert math.isfinite(outcome["variance_qlike_unnormalized"])


def test_adverse_holdout_negative_r2_and_proper_scores_are_retained(fitted, datasets):
    # Deliberately hostile SYNTHETIC labels test reporting, not model benefit.
    holdout = tuple(
        replace(r, log_return=-fitted.predict(r.observation).mean_log_return) for r in datasets[2]
    )
    result = fitted.evaluate(holdout, asof=START + timedelta(days=90))
    assert result["outcomes"][0]["return_r2_diagnostic"] <= -3
    assert len(result["outcomes"]) == 2 and result["synthetic"]
    assert not result["market_evidence"]


@pytest.mark.parametrize(
    "fault", ["fit_overlap", "holdout_overlap", "event_overlap", "embargo", "late_eval"]
)
def test_disjoint_heldout_evidence_and_embargo(datasets, fitted, fault):
    train, calibration, holdout = datasets
    if fault == "fit_overlap":
        with pytest.raises(ValueError):
            EconomicEventModel.fit(
                train,
                train[:16],
                config=CONFIG,
                train_asof=START + timedelta(days=35),
                calibration_asof=START + timedelta(days=60),
            )
    elif fault == "holdout_overlap":
        with pytest.raises(ValueError, match="calibration"):
            fitted.evaluate(calibration, asof=START + timedelta(days=90))
    elif fault == "event_overlap":
        observation = holdout[0].observation
        oldid = train[0].observation.calendar.event_id
        observation = replace(
            observation,
            calendar=replace(observation.calendar, event_id=oldid),
            releases=tuple(replace(r, event_id=oldid) for r in observation.releases),
            consensus=tuple(replace(r, event_id=oldid) for r in observation.consensus),
        )
        with pytest.raises(ValueError, match="overlap"):
            fitted.predict(observation)
    elif fault == "embargo":
        with pytest.raises(ValueError, match="embargo"):
            EconomicEventModel.fit(
                train,
                calibration,
                config=replace(CONFIG, embargo_seconds=86400),
                train_asof=START + timedelta(days=39, hours=20),
                calibration_asof=START + timedelta(days=60),
            )
    else:
        with pytest.raises(ValueError, match="not yet published"):
            fitted.evaluate(holdout, asof=START + timedelta(days=80))


def test_delay_masking_is_applied_before_feature_preprocessing(datasets):
    train, calibration, _ = datasets
    with pytest.raises(ValueError, match="not yet delivered"):
        EconomicEventModel.fit(
            train,
            calibration,
            config=replace(CONFIG, release_delay_seconds=3),
            train_asof=START + timedelta(days=35),
            calibration_asof=START + timedelta(days=60),
        )


def test_incoherent_manual_return_variance_and_unseen_declared_kind_rejected(datasets):
    train, calibration, _ = datasets
    impossible = replace(train[0], log_return=0.01, realized_variance=0.0)
    with pytest.raises(ValueError, match="incoherent"):
        EconomicEventModel.fit(
            (impossible, *train[1:]),
            calibration,
            config=CONFIG,
            train_asof=START + timedelta(days=35),
            calibration_asof=START + timedelta(days=60),
        )
    too_little_variance = replace(train[0], log_return=0.01, realized_variance=0.01**2 / 13)
    with pytest.raises(ValueError, match="incoherent"):
        EconomicEventModel.fit(
            (too_little_variance, *train[1:]),
            calibration,
            config=CONFIG,
            train_asof=START + timedelta(days=35),
            calibration_asof=START + timedelta(days=60),
        )
    with pytest.raises(ValueError, match="observed training"):
        EconomicEventModel.fit(
            train,
            calibration,
            config=replace(CONFIG, kinds=("cpi_yoy", "cpi_mom")),
            train_asof=START + timedelta(days=35),
            calibration_asof=START + timedelta(days=60),
        )
    with pytest.raises(ValueError, match="consensus unavailable"):
        EconomicEventModel.fit(
            train,
            calibration,
            config=replace(CONFIG, consensus_delay_seconds=4000),
            train_asof=START + timedelta(days=35),
            calibration_asof=START + timedelta(days=60),
        )


def postprices(observation, *, synthetic=True):
    return tuple(
        TimedPrice(
            f"post_{i}",
            observation.decision_at + timedelta(minutes=i),
            observation.decision_at + timedelta(minutes=i, seconds=1),
            100 * math.exp(0.0005 * i),
            SHA,
            synthetic,
        )
        for i in range(61)
    )


def test_spike_threshold_uses_training_past_prices_and_only_observed_postprices(fitted, datasets):
    observation = datasets[2][0].observation
    prices = postprices(observation)
    baseline = fitted.metadata()["spike_threshold"]
    expected = np.quantile(
        np.abs(
            [value for r in fitted._body["training"] for value in r["snapshot"]["past_returns"]]
        ),
        CONFIG.spike_quantile,
    )
    assert baseline == pytest.approx(max(expected, CONFIG.min_sigma))
    early = fitted.detect_spikes(
        observation, prices, asof=observation.decision_at + timedelta(minutes=30, seconds=2)
    )
    late = fitted.detect_spikes(
        observation, prices, asof=observation.decision_at + timedelta(hours=1, seconds=2)
    )
    assert early["spike"] and not early["complete_horizon"]
    assert late["complete_horizon"] and late["spike"]
    assert early["visible_price_sha256"] != late["visible_price_sha256"]
    assert fitted.metadata()["spike_threshold"] == baseline
    assert not late["causal_effect_identified"]
    with pytest.raises(ValueError, match="anchor"):
        fitted.detect_spikes(observation, prices, asof=observation.decision_at)
    with pytest.raises(ValueError, match="same declared interval"):
        fitted.detect_spikes(
            observation, prices[::2], asof=observation.decision_at + timedelta(hours=1, seconds=2)
        )


def test_actual_market_reaction_measurement_binds_complete_timed_sampling_grid(datasets):
    observation = datasets[2][0].observation
    start = observation.decision_at
    prices = tuple(
        TimedPrice(
            f"reaction{i}",
            start + timedelta(seconds=300 * i),
            start + timedelta(seconds=300 * i + 2),
            100 * math.exp(0.0001 * i * i),
            SHA,
            True,
        )
        for i in range(13)
    )
    measured = measure_reaction(
        observation, prices, config=CONFIG, asof=start + timedelta(hours=1, seconds=3)
    )
    expected_returns = np.diff(np.log([p.price for p in prices]))
    assert measured.log_return == pytest.approx(
        math.log(prices[-1].price / prices[0].price), abs=1e-14
    )
    assert measured.realized_variance == pytest.approx(np.sum(expected_returns**2), abs=1e-14)
    assert measured.available_at == prices[-1].available_at
    assert measured.source_sha256 == _hash(
        [
            {
                "record_id": p.record_id,
                "timestamp": p.timestamp,
                "available_at": p.available_at,
                "price": p.price,
                "source_sha256": p.source_sha256,
                "synthetic": p.synthetic,
            }
            for p in prices
        ]
    )
    assert measured.synthetic
    with pytest.raises(ValueError, match="sampling grid"):
        measure_reaction(
            observation, prices[:-1], config=CONFIG, asof=start + timedelta(hours=1, seconds=3)
        )
    with pytest.raises(ValueError, match="not yet delivered"):
        measure_reaction(observation, prices, config=CONFIG, asof=start + timedelta(hours=1))
    with pytest.raises(ValueError, match="not yet delivered"):
        measure_reaction(
            observation,
            prices,
            config=replace(CONFIG, price_delay_seconds=5),
            asof=start + timedelta(hours=1, seconds=3),
        )


def test_source_labels_synthetic_or_propagation_and_numeric_domain():
    with pytest.raises(ValueError, match="SHA256"):
        replace(row(1).observation.releases[0], source_sha256="fake")
    with pytest.raises(ValueError, match="Boolean"):
        replace(row(1).observation.releases[0], synthetic=1)
    with pytest.raises(ValueError, match="nonnegative"):
        replace(row(1), realized_variance=-1)
    with pytest.raises(ValueError, match="finite"):
        replace(row(1), log_return=float("nan"))
    train = tuple(row(i, synthetic=False) for i in range(32))
    calibration = tuple(row(i, synthetic=False) for i in range(40, 56))
    fitted = EconomicEventModel.fit(
        train,
        calibration,
        config=CONFIG,
        train_asof=START + timedelta(days=35),
        calibration_asof=START + timedelta(days=60),
    )
    assert not fitted.metadata()["synthetic"]
    heldout = tuple(row(i, synthetic=False) for i in range(70, 86))
    heldout = (replace(heldout[0], synthetic=True), *heldout[1:])
    assert fitted.evaluate(heldout, asof=START + timedelta(days=90))["synthetic"]
    assert not fitted.metadata()["authenticated_sources"]


def test_safe_json_write_once_and_fitting_replay(fitted, datasets, tmp_path):
    path = tmp_path / "model.json"
    digest = fitted.save(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        fitted.save(path)
    loaded = EconomicEventModel.load(path)
    assert loaded.model_sha256 == fitted.model_sha256
    assert loaded.predict(datasets[2][0].observation) == fitted.predict(datasets[2][0].observation)
    assert loaded.evaluate(datasets[2], asof=START + timedelta(days=90)) == fitted.evaluate(
        datasets[2], asof=START + timedelta(days=90)
    )


@pytest.mark.parametrize(
    "fault",
    [
        "weights",
        "scaler",
        "spike",
        "features",
        "price",
        "cal_time",
        "flag",
        "source",
        "rows",
        "unknown_key",
    ],
)
def test_rehashed_json_forgery_cannot_bypass_numeric_or_causal_replay(fitted, tmp_path, fault):
    path = tmp_path / "bad.json"
    fitted.save(path)
    payload = json.loads(path.read_text())
    body = payload["body"]
    if fault == "weights":
        body["learned"]["arms"]["surprise"]["beta"][0] += 1
    elif fault == "scaler":
        body["learned"]["scale"][0] = 0
    elif fault == "spike":
        body["learned"]["spike_threshold"] *= 10
    elif fault == "features":
        body["training"][0]["snapshot"]["features"][0] += 1
    elif fault == "price":
        body["training"][0]["snapshot"]["past_prices"][0]["price"] *= 2
    elif fault == "cal_time":
        body["calibration_asof"] = body["train_asof"]
    elif fault == "flag":
        body["market_evidence"] = True
    elif fault == "source":
        body["implementation_sha256"] = "0" * 64
    elif fault == "rows":
        body["training"] *= 200
    elif fault == "unknown_key":
        body["lie"] = True
    payload["model_sha256"] = _hash(body)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        EconomicEventModel.load(path)


@pytest.mark.parametrize("fault", ["kinds", "trainclock", "decisionclock", "targetclock"])
def test_malformed_resealed_snapshot_types_rejected_as_validation_errors(fitted, tmp_path, fault):
    path = tmp_path / "malformed.json"
    fitted.save(path)
    payload = json.loads(path.read_text())
    body = payload["body"]
    if fault == "kinds":
        body["config"]["kinds"] = None
    elif fault == "trainclock":
        body["train_asof"] = 123
    elif fault == "decisionclock":
        body["training"][0]["snapshot"]["decision_at"] = 123
    else:
        body["training"][0]["target_end"] = None
    payload["model_sha256"] = _hash(body)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema"):
        EconomicEventModel.load(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon_seconds": 0},
        {"horizon_seconds": True},
        {"ridge_alpha": 0},
        {"kinds": ("unknown",)},
        {"kinds": ("cpi_yoy", "cpi_yoy")},
        {"spike_quantile": 1.0},
        {"price_delay_seconds": -1},
    ],
)
def test_bounded_config_fail_closed(kwargs):
    with pytest.raises(ValueError):
        EventConfig("SYNTHETIC_SPY", **kwargs)


def test_model_mutation_unknown_kind_and_bounded_json_fail_closed(fitted, datasets, tmp_path):
    changed = replace(datasets[2][0].observation, asset_id="unknown")
    with pytest.raises(ValueError, match="asset"):
        fitted.predict(changed)
    with pytest.raises(ValueError, match="kind"):
        observation = datasets[2][0].observation
        changed = replace(observation, calendar=replace(observation.calendar, kind="cpi_mom"))
        fitted.predict(changed)
    fitted._body["learned"]["arms"]["surprise"]["sigma"] = 99
    with pytest.raises(ValueError, match="seal"):
        fitted.predict(datasets[2][0].observation)
    path = Path(tmp_path) / "oversized.json"
    path.write_bytes(b" " * 16000001)
    with pytest.raises(ValueError, match="resource"):
        EconomicEventModel.load(path)
