"""SYNTHETIC causal/learning/replay/rollback checks, not market evidence."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from quant_fund.models.continual_market import (
    ContinualConfig,
    ContinualForecast,
    ContinualMarket,
    ContinualSnapshot,
    MarketObservation,
    MarketTarget,
    _hash,
    _new_statistics,
    _optimize,
    _scales,
)

BASE = datetime(2020, 1, 1, tzinfo=UTC)
EVIDENCE = "a" * 64


def _record(i, features=(0.5, -0.25), y=None, synthetic=True):
    event = BASE + timedelta(hours=i)
    target_event = event + timedelta(minutes=5)
    obs = MarketObservation(
        str(i),
        "ASSET",
        ("x1", "x2"),
        tuple(features),
        event,
        event + timedelta(minutes=1),
        target_event,
        "synthetic_feature_fixture",
        EVIDENCE,
        synthetic,
    )
    target = MarketTarget(
        str(i),
        1.2 * features[0] - 0.8 * features[1] + 0.3 if y is None else y,
        target_event,
        event + timedelta(minutes=8),
        "synthetic_target_fixture",
        EVIDENCE,
        synthetic,
    )
    return obs, target


def _step(model, pair):
    observation, target = pair
    prediction = model.predict(
        observation, decision_time=observation.available_time + timedelta(minutes=1)
    )
    update = model.observe(target, as_of=target.available_time)
    assert update.forecast == prediction
    return update


def _stream(n=80, seed=114, start=0, shift=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    noise = rng.normal(0, 0.2, n)
    return tuple(
        _record(
            i + start, tuple(features), 1.2 * features[0] - 0.8 * features[1] + 0.3 + shift + error
        )
        for i, (features, error) in enumerate(zip(x, noise, strict=True))
    )


def _warm(model, n=40):
    pairs = _stream(n)
    for pair in pairs:
        _step(model, pair)
    return pairs


def test_cold_start_never_fabricates_confidence_or_scores():
    model = ContinualMarket(("x1", "x2"))
    obs, target = _record(0)
    forecast = model.predict(obs, decision_time=obs.available_time)
    assert forecast.status == "cold_start" and forecast.mean is None and forecast.scale is None
    assert forecast.model_sha256 is None
    with pytest.raises(RuntimeError, match="cold-start"):
        forecast.score(0)
    with pytest.raises(RuntimeError, match="cold start"):
        model.freeze()
    result = model.observe(target, as_of=target.available_time)
    assert result.scores is None and model.n_training == 1
    assert result.synthetic and result.research_only and not result.market_evidence


@pytest.mark.parametrize(
    "fault",
    [
        "event_naive",
        "publication_naive",
        "target_naive",
        "publication_before",
        "target_not_future",
        "bad_hash",
        "bad_names",
        "nonfinite",
        "nonboolean",
    ],
)
def test_observation_validation(fault):
    obs, _ = _record(0)
    changes = {
        "event_naive": {"event_time": obs.event_time.replace(tzinfo=None)},
        "publication_naive": {"available_time": obs.available_time.replace(tzinfo=None)},
        "target_naive": {"target_event_time": obs.target_event_time.replace(tzinfo=None)},
        "publication_before": {"available_time": obs.event_time - timedelta(seconds=1)},
        "target_not_future": {"target_event_time": obs.event_time},
        "bad_hash": {"evidence_sha256": "abc"},
        "bad_names": {"feature_names": ("x", "x")},
        "nonfinite": {"features": (float("nan"), 0)},
        "nonboolean": {"synthetic": 1},
    }
    with pytest.raises(ValueError):
        replace(obs, **changes[fault])


@pytest.mark.parametrize(
    "fault", ["event_naive", "publication_before", "nonfinite", "boolean_value", "bad_hash"]
)
def test_target_validation(fault):
    _, target = _record(0)
    changes = {
        "event_naive": {"event_time": target.event_time.replace(tzinfo=None)},
        "publication_before": {"available_time": target.event_time - timedelta(seconds=1)},
        "nonfinite": {"value": float("inf")},
        "boolean_value": {"value": True},
        "bad_hash": {"evidence_sha256": "b"},
    }
    with pytest.raises(ValueError):
        replace(target, **changes[fault])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"updates_per_label": 0},
        {"reservoir_capacity": 513},
        {"replay_batch_size": 129},
        {"min_labels": 3},
        {"drift_window": 3},
        {"max_pending": 513},
        {"learning_rate": 0.2},
        {"momentum": 1},
        {"gradient_clip": float("nan")},
        {"initial_scale": 0},
        {"drift_learning_rate_multiplier": 10},
    ],
)
def test_resource_and_optimizer_bounds(kwargs):
    with pytest.raises(ValueError):
        ContinualConfig(**kwargs)


def test_unpublished_features_future_labels_and_clock_regression_leave_state_unchanged():
    model = ContinualMarket(("x1", "x2"))
    obs, target = _record(0)
    initial = model.snapshot()
    with pytest.raises(ValueError, match="published"):
        model.predict(obs, decision_time=obs.event_time)
    assert model.snapshot() == initial
    forecast = model.predict(obs, decision_time=obs.available_time)
    pending = model.snapshot()
    with pytest.raises(ValueError, match="not published"):
        model.observe(target, as_of=target.event_time)
    assert model.snapshot() == pending and model.n_training == 0
    with pytest.raises(ValueError, match="backwards"):
        model.predict(_record(-1)[0], decision_time=BASE - timedelta(minutes=1))
    assert model.snapshot() == pending
    update = model.observe(target, as_of=target.available_time)
    assert update.forecast.prediction_sha256 == forecast.prediction_sha256


def test_wrong_feature_order_target_endpoint_duplicate_prediction_or_delivery_rejected():
    model = ContinualMarket(("x1", "x2"))
    obs, target = _record(0)
    with pytest.raises(ValueError, match="names/order"):
        model.predict(replace(obs, feature_names=("x2", "x1")), decision_time=obs.available_time)
    with pytest.raises(ValueError, match="original"):
        model.observe(target, as_of=target.available_time)
    model.predict(obs, decision_time=obs.available_time)
    before = model.snapshot()
    with pytest.raises(ValueError, match="already predicted"):
        model.predict(obs, decision_time=obs.available_time)
    with pytest.raises(ValueError, match="endpoint"):
        model.observe(
            replace(target, event_time=target.event_time + timedelta(minutes=1)),
            as_of=target.available_time,
        )
    assert model.snapshot() == before
    model.observe(target, as_of=target.available_time)
    with pytest.raises(ValueError, match="unconsumed"):
        model.observe(target, as_of=target.available_time)
    assert model.n_training == 1


def test_pending_predictions_do_not_update_scaling_or_see_delayed_labels():
    model = ContinualMarket(("x1", "x2"))
    pairs = _stream(10)
    forecasts = [model.predict(obs, decision_time=obs.available_time) for obs, _ in pairs]
    assert model.n_training == 0
    state = json.loads(model.snapshot().state_json)
    assert state["preprocessing"]["count"] == 0 and state["preprocessing"]["mean"] == [0, 0]
    delivery = BASE + timedelta(hours=11)
    for i, (_, target) in enumerate(pairs):
        result = model.observe(target, as_of=delivery)
        assert result.forecast == forecasts[i] and result.scores is None
    assert model.n_training == 10
    assert all(f.status == "cold_start" for f in forecasts)


def test_original_fitted_prediction_is_preserved_while_other_deliveries_change_model():
    model = ContinualMarket(("x1", "x2"))
    _warm(model, 30)
    obs, target = _record(31, (1.3, -0.8))
    prediction = model.predict(obs, decision_time=obs.available_time)
    _step(model, _record(32, (3.0, -1), 10.0))
    update = model.observe(target, as_of=BASE + timedelta(hours=33))
    assert update.forecast == prediction
    assert update.scores == prediction.score(target.value)
    assert update.forecast.model_sha256 != model.freeze().model_sha256


def test_training_statistics_count_unique_deliveries_once_not_replay_updates():
    model = ContinualMarket(
        ("x1", "x2"), ContinualConfig(updates_per_label=8, replay_batch_size=16)
    )
    pairs = _warm(model, 70)
    x = np.asarray([obs.features for obs, _ in pairs])
    state = json.loads(model.snapshot().state_json)
    assert state["preprocessing"]["count"] == len(pairs) == model.n_training
    assert model.optimizer_steps == len(pairs) * 8
    np.testing.assert_allclose(state["preprocessing"]["mean"], x.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(
        state["preprocessing"]["m2"], ((x - x.mean(axis=0)) ** 2).sum(axis=0), atol=1e-12
    )


def test_scaler_rebasing_preserves_raw_predictions_and_momentum_deltas():
    model = ContinualMarket(("x1", "x2"))
    _warm(model, 20)
    state = json.loads(model.snapshot().state_json)
    probes = np.asarray([[0.2, 0.7], [-4.0, 8.0], [1.0, 0]])
    old_x = (probes - state["preprocessing"]["mean"]) / _scales(state, model.config)
    old_mean = old_x @ state["parameters"]["coef"] + state["parameters"]["intercept"]
    old_velocity = np.asarray(state["optimizer"]["velocity"])
    old_delta = old_x @ old_velocity[:-2] + old_velocity[-2]
    _new_statistics(state, np.asarray([100.0, -40.0]), model.config)
    new_x = (probes - state["preprocessing"]["mean"]) / _scales(state, model.config)
    new_mean = new_x @ state["parameters"]["coef"] + state["parameters"]["intercept"]
    new_velocity = np.asarray(state["optimizer"]["velocity"])
    new_delta = new_x @ new_velocity[:-2] + new_velocity[-2]
    np.testing.assert_allclose(new_mean, old_mean, atol=1e-13)
    np.testing.assert_allclose(new_delta, old_delta, atol=1e-13)


def test_gaussian_nll_analytic_gradient_matches_independent_finite_differences():
    config = ContinualConfig(learning_rate=1e-4, updates_per_label=1, momentum=0, gradient_clip=1e6)
    state = json.loads(ContinualMarket(("x1", "x2"), config).snapshot().state_json)
    state["preprocessing"] = {"count": 8, "mean": [0.0, 0.0], "m2": [7.0, 7.0]}
    initial = np.asarray([0.2, -0.3, 0.1, math.log(0.8)])
    state["parameters"] = {
        "coef": initial[:2].tolist(),
        "intercept": initial[2],
        "log_variance": initial[3],
    }
    x, y = np.asarray([[0.1, 0.2], [1.0, -0.2], [-0.5, 0.7]]), np.asarray([0.3, 0.9, -0.2])
    rows = [
        {"observation": {"features": list(a)}, "target": {"value": float(b)}}
        for a, b in zip(x, y, strict=True)
    ]

    def loss(params):
        residual = x @ params[:2] + params[2] - y
        return float(
            0.5 * np.mean(residual**2) / math.exp(params[3])
            + 0.5 * params[3]
            + 0.5 * math.log(2 * math.pi)
        )

    gradient = []
    for i in range(4):
        plus, minus = initial.copy(), initial.copy()
        plus[i] += 1e-6
        minus[i] -= 1e-6
        gradient.append((loss(plus) - loss(minus)) / 2e-6)
    _optimize(state, rows, config, config.learning_rate)
    actual = np.asarray(
        state["parameters"]["coef"]
        + [state["parameters"]["intercept"], state["parameters"]["log_variance"]]
    )
    np.testing.assert_allclose(
        actual, initial - config.learning_rate * np.asarray(gradient), rtol=0, atol=1e-12
    )


def test_gaussian_proper_scores_against_independent_cdf_integral():
    from scipy.integrate import quad
    from scipy.special import ndtr

    forecast = ContinualForecast(
        "score",
        BASE,
        BASE + timedelta(hours=1),
        "forecast",
        0.3,
        0.7,
        EVIDENCE,
        EVIDENCE,
        EVIDENCE,
        True,
    )
    target = -0.2
    scores = forecast.score(target)
    assert scores.gaussian_nll == pytest.approx(
        math.log(0.7 * math.sqrt(2 * math.pi)) + ((target - 0.3) / 0.7) ** 2 / 2
    )
    expected = (
        quad(lambda x: float(ndtr((x - 0.3) / 0.7)) ** 2, -np.inf, target)[0]
        + quad(lambda x: (1 - float(ndtr((x - 0.3) / 0.7))) ** 2, target, np.inf)[0]
    )
    assert scores.gaussian_crps == pytest.approx(expected, abs=1e-10)


def test_frozen_forecast_and_model_identity_mutation_is_rejected():
    model = ContinualMarket(("x1", "x2"))
    _warm(model, 12)
    frozen = model.freeze()
    obs, _ = _record(13)
    forecast = frozen.predict(obs, decision_time=obs.available_time)
    object.__setattr__(forecast, "mean", 100.0)
    with pytest.raises(RuntimeError, match="forecast changed"):
        forecast.score(0)
    object.__setattr__(frozen, "intercept", 100.0)
    with pytest.raises(RuntimeError, match="identity changed"):
        frozen.predict(obs, decision_time=obs.available_time)


@pytest.mark.parametrize(
    "fault",
    [
        "negative_variance",
        "unpublished_replay",
        "duplicate_delivered",
        "fit_after_clock",
        "implementation",
        "runtime",
    ],
)
def test_forged_snapshot_semantics_are_validated_before_rollback(tmp_path, fault):
    model = ContinualMarket(("x1", "x2"))
    _warm(model, 12)
    original = model.snapshot()
    state = json.loads(original.state_json)
    if fault == "negative_variance":
        state["preprocessing"]["m2"][0] = -10
    elif fault == "unpublished_replay":
        state["reservoir"][0]["target"]["available_time"] = (
            BASE + timedelta(hours=100)
        ).isoformat()
    elif fault == "duplicate_delivered":
        state["delivered"][1] = state["delivered"][0]
    elif fault == "fit_after_clock":
        state["fit_as_of"] = (BASE + timedelta(hours=100)).isoformat()
    elif fault == "implementation":
        state["implementation_sha256"] = "f" * 64
    elif fault == "runtime":
        state["numpy_version"] = "different"
    forged = ContinualSnapshot(json.dumps(state), _hash(state))
    with pytest.raises(ValueError):
        model.rollback(forged)
    assert model.snapshot() == original


def test_failed_optimizer_update_does_not_consume_prediction_or_rng(monkeypatch):
    import quant_fund.models.continual_market as module

    model = ContinualMarket(("x1", "x2"))
    _warm(model, 12)
    obs, target = _record(13)
    model.predict(obs, decision_time=obs.available_time)
    original = model.snapshot()

    def fail(state, *args):
        state["parameters"]["intercept"] = 100
        raise FloatingPointError("injected optimizer failure")

    monkeypatch.setattr(module, "_optimize", fail)
    with pytest.raises(FloatingPointError):
        model.observe(target, as_of=target.available_time)
    assert model.snapshot() == original


def test_learned_prequential_improvement_on_independently_generated_synthetic_stream():
    model = ContinualMarket(("x1", "x2"))
    training = _stream(400, seed=813)
    updates = [_step(model, pair) for pair in training[:8]]
    early_model = model.freeze()
    updates.extend(_step(model, pair) for pair in training[8:])
    late_model = model.freeze()
    # The same separately generated, never-trained holdout distinguishes
    # learning from differences between two noisy training-stream slices.
    holdout = _stream(1024, seed=999, start=500)
    early = [
        early_model.predict(obs, decision_time=obs.available_time).score(target.value)
        for obs, target in holdout
    ]
    late = [
        late_model.predict(obs, decision_time=obs.available_time).score(target.value)
        for obs, target in holdout
    ]
    assert np.mean([v.gaussian_crps for v in late]) < np.mean([v.gaussian_crps for v in early])
    assert np.mean([v.gaussian_nll for v in late]) < np.mean([v.gaussian_nll for v in early])
    assert all(v.synthetic and not v.market_evidence for v in updates)
    frozen = model.freeze()
    raw_coefficients = np.asarray(frozen.coefficients) / frozen.feature_scale
    np.testing.assert_allclose(raw_coefficients, [1.2, -0.8], atol=0.25)
    assert frozen.scale < 0.7


def test_prefix_forecasts_and_snapshots_do_not_depend_on_unpublished_future_suffix():
    one, two = ContinualMarket(("x1", "x2")), ContinualMarket(("x1", "x2"))
    pairs = _stream(50)
    for pair in pairs[:40]:
        assert _step(one, pair) == _step(two, pair)
    assert one.snapshot() == two.snapshot()
    obs, target = pairs[40]
    future = replace(target, value=1000.0)
    p1 = one.predict(obs, decision_time=obs.available_time)
    p2 = two.predict(obs, decision_time=obs.available_time)
    assert p1 == p2
    before = two.snapshot()
    with pytest.raises(ValueError, match="not published"):
        two.observe(future, as_of=target.event_time)
    assert two.snapshot() == before and one.snapshot() == two.snapshot()


def test_reservoir_is_seeded_bounded_and_stores_published_examples():
    config = ContinualConfig(reservoir_capacity=7, replay_batch_size=4)
    one, two = ContinualMarket(("x1", "x2"), config), ContinualMarket(("x1", "x2"), config)
    for pair in _stream(100):
        _step(one, pair)
        _step(two, pair)
    assert one.reservoir_ids == two.reservoir_ids and len(one.reservoir_ids) == 7
    assert len(set(one.reservoir_ids)) == 7
    assert one.reservoir_ids != tuple(str(i) for i in range(7))
    assert one.snapshot() == two.snapshot()
    state = json.loads(one.snapshot().state_json)
    for row in state["reservoir"]:
        assert row["target"]["record_id"] in state["delivered"]
        assert datetime.fromisoformat(row["target"]["available_time"]) <= datetime.fromisoformat(
            state["clock"]
        )


def test_snapshot_exact_rollback_restores_pending_forecasts_optimizer_rng_and_provenance():
    model = ContinualMarket(
        ("x1", "x2"), ContinualConfig(reservoir_capacity=9, replay_batch_size=5)
    )
    _warm(model, 40)
    pending = _record(41)
    forecast = model.predict(pending[0], decision_time=pending[0].available_time)
    checkpoint = model.snapshot()
    initial_delivery = model.observe(pending[1], as_of=pending[1].available_time)
    suffix = _stream(20, start=42, seed=55)
    first = [_step(model, pair) for pair in suffix]
    final = model.snapshot()
    model.rollback(checkpoint)
    assert model.snapshot() == checkpoint
    repeated = model.observe(pending[1], as_of=pending[1].available_time)
    assert repeated == initial_delivery and repeated.forecast == forecast
    second = [_step(model, pair) for pair in suffix]
    assert first == second and model.snapshot() == final


def test_snapshot_serialization_restores_into_new_instance_and_detects_tampering():
    model = ContinualMarket(("x1", "x2"))
    _warm(model, 20)
    snapshot = model.snapshot()
    restored = ContinualMarket(("different",), ContinualConfig(seed=10))
    restored.rollback(ContinualSnapshot(snapshot.state_json, snapshot.sha256))
    assert restored.snapshot() == snapshot and restored.freeze() == model.freeze()
    with pytest.raises(ValueError, match="digest"):
        ContinualSnapshot(snapshot.state_json + " ", "b" * 64)
    state = json.loads(snapshot.state_json)
    state["preprocessing"]["count"] += 1
    image = json.dumps(state)
    forged = ContinualSnapshot(image, _hash(state))
    with pytest.raises(ValueError, match="count"):
        restored.rollback(forged)
    assert restored.snapshot() == snapshot


def test_mutated_state_or_config_fails_closed_and_valid_rollback_recovers():
    model = ContinualMarket(("x1", "x2"))
    _warm(model, 20)
    checkpoint = model.snapshot()
    model._state["parameters"]["coef"][0] += 1
    with pytest.raises(RuntimeError, match="changed"):
        model.freeze()
    model.rollback(checkpoint)
    assert model.snapshot() == checkpoint
    model.config = replace(model.config, learning_rate=0.01)
    with pytest.raises(RuntimeError, match="changed"):
        model.freeze()
    model.rollback(checkpoint)
    assert model.snapshot() == checkpoint


def test_freeze_is_immutable_pit_safe_and_independent_of_later_updates():
    model = ContinualMarket(("x1", "x2"))
    _warm(model, 30)
    frozen = model.freeze()
    obs, target = _record(31)
    forecast = frozen.predict(obs, decision_time=obs.available_time)
    with pytest.raises(FrozenInstanceError):
        frozen.intercept = 10
    _step(model, (obs, target))
    assert frozen.predict(obs, decision_time=obs.available_time) == forecast
    with pytest.raises(ValueError, match="fitted after"):
        frozen.predict(_record(2)[0], decision_time=_record(2)[0].available_time)


def test_drift_uses_only_published_prequential_losses_and_bounded_current_update_adaptation():
    config = ContinualConfig(
        drift_window=4, drift_threshold=0.1, drift_learning_rate_multiplier=2.0
    )
    model = ContinualMarket(("x1", "x2"), config)
    _warm(model, 60)
    observation, target = _record(61, (0.2, 0.1), 20.0)
    model.predict(observation, decision_time=observation.available_time)
    before = model.snapshot()
    old_drift = model.drift_status
    with pytest.raises(ValueError, match="not published"):
        model.observe(target, as_of=target.event_time)
    assert model.drift_status == old_drift and model.snapshot() == before
    shifted = [model.observe(target, as_of=target.available_time)]
    shifted.extend(_step(model, pair) for pair in _stream(12, start=62, shift=15, seed=18))
    assert any(v.drift_alert for v in shifted)
    for update in shifted:
        assert update.effective_learning_rate == config.learning_rate * (
            2 if update.drift_alert else 1
        )
        assert update.effective_learning_rate <= 0.1
    assert model.n_training == 73 and len(model.reservoir_ids) == 73
    assert model.optimizer_steps == 73 * config.updates_per_label


def test_forgetting_audit_compares_frozen_learned_batch_baseline_without_adaptation():
    model = ContinualMarket(("x1", "x2"))
    prefix = _warm(model, 40)
    initial = json.loads(model.snapshot().state_json)["baseline"]
    for pair in _stream(20, start=41, shift=2, seed=8):
        _step(model, pair)
    checkpoint = model.snapshot()
    report = model.audit_forgetting(prefix[:8], as_of=BASE + timedelta(hours=70))
    assert model.snapshot() == checkpoint
    assert report["baseline_model_sha256"] == initial["model_sha256"]
    assert report["n_reused_training_ids"] == 8 and not report["independent_holdout_established"]
    assert report["status"] == "retrospective_score_only_forgetting_audit"
    assert report["synthetic"] and not report["market_evidence"]
    assert report["current_minus_baseline_nll"] == pytest.approx(
        report["current_gaussian_nll"] - report["baseline_gaussian_nll"]
    )
    unknown = _record(100)
    with pytest.raises(ValueError, match="already-published"):
        model.audit_forgetting((prefix[0], unknown), as_of=BASE + timedelta(hours=70))
    assert model.snapshot() == checkpoint


def test_pending_and_stream_resource_bounds_are_fail_closed():
    model = ContinualMarket(("x1", "x2"), ContinualConfig(max_pending=1))
    one, two = _record(0), _record(1)
    model.predict(one[0], decision_time=one[0].available_time)
    before = model.snapshot()
    with pytest.raises(ValueError, match="resource bound"):
        model.predict(two[0], decision_time=two[0].available_time)
    assert model.snapshot() == before


def test_synthetic_training_provenance_is_not_dropped_for_empirical_flagged_prediction():
    model = ContinualMarket(("x1", "x2"))
    _warm(model, 20)
    observation, target = _record(21, synthetic=False)
    forecast = model.predict(observation, decision_time=observation.available_time)
    update = model.observe(target, as_of=target.available_time)
    assert forecast.synthetic and update.synthetic
    assert not forecast.market_evidence and not update.market_evidence
    assert model.freeze().training_synthetic
