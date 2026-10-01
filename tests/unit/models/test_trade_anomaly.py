"""Synthetic anomaly correctness, never insider-identification evidence."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from quant_fund.models.trade_anomaly import (
    PatternAnnotation,
    TradeAnomalyModel,
    TradePattern,
    _hash,
)

START = datetime(2020, 1, 1, tzinfo=UTC)


def patterns(first, n, *, unusual=False, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for i in range(first, first + n):
        event = START + timedelta(minutes=i)
        anomaly = unusual and i % 2 == 0
        size = float(rng.normal(15 if anomaly else 5, 0.2))
        row = TradePattern(
            str(i),
            event,
            event + timedelta(seconds=1),
            event + timedelta(seconds=2),
            (
                size,
                0.01,
                float(rng.normal()),
                float(rng.normal(10 if anomaly else 0, 0.2)),
                float(rng.uniform(1, 10)),
                float(rng.uniform()),
            ),
            "SYNTHETIC_trades",
            True,
        )
        rows.append(row)
        labels.append(
            PatternAnnotation(
                str(i),
                anomaly,
                event + timedelta(seconds=3),
                "SYNTHETIC_independent_injection_log",
                "predeclared injected event",
                True,
            )
        )
    return rows, labels


@pytest.fixture
def fitted():
    rows, _ = patterns(0, 128)
    return TradeAnomalyModel(trees=16, samples=64).fit(rows, asof=START + timedelta(minutes=128))


@pytest.fixture
def calibrated(fitted):
    rows, labels = patterns(129, 96, unusual=True)
    return fitted.calibrate(rows, labels, asof=START + timedelta(minutes=225))


def test_actual_isolation_forest_matches_independent_sklearn_scores(fitted):
    train, _ = patterns(0, 128)
    evaluation, _ = patterns(226, 80, unusual=True, seed=19)
    reference = IsolationForest(
        n_estimators=16,
        max_samples=64,
        max_features=1.0,
        contamination="auto",
        random_state=7,
        n_jobs=1,
    ).fit(np.array([r.features for r in train], dtype=np.float32))
    expected = -reference.score_samples(
        np.array([r.features for r in evaluation], dtype=np.float32)
    )
    actual = fitted.predict(evaluation)
    np.testing.assert_allclose([r.anomaly_score for r in actual], expected, atol=1e-14, rtol=1e-14)
    assert all(r.suspicious_annotation_probability is None for r in actual)
    assert all(r.intent == r.insider_misconduct == "UNKNOWN" for r in actual)
    assert np.mean(expected[::2]) > np.mean(expected[1::2])


def test_actual_calibration_and_proper_scores_against_independent_formula(calibrated):
    rows, labels = patterns(226, 80, unusual=True, seed=19)
    result = calibrated.evaluate(rows, labels, asof=START + timedelta(minutes=307))
    p = np.array([r.suspicious_annotation_probability for r in calibrated.predict(rows)])
    y = np.array([a.suspicious for a in labels], dtype=float)
    assert result["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert result["log_loss"] == pytest.approx(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))
    assert result["brier"] < result["baseline_brier"]
    alerts = p >= 0.5
    tp = np.sum(alerts & (y == 1))
    fp = np.sum(alerts & (y == 0))
    fn = np.sum(~alerts & (y == 1))
    assert result["diagnostics"]["precision"] == pytest.approx(tp / (tp + fp))
    assert result["diagnostics"]["recall"] == pytest.approx(tp / (tp + fn))
    assert result["diagnostics"]["f1"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert result["synthetic"] and not result["market_evidence"]
    assert not result["annotation_independence_verified"]


def test_missing_annotations_never_produces_probability_or_pass(fitted):
    rows, _ = patterns(129, 64)
    assert fitted.predict(rows)[0].suspicious_annotation_probability is None
    with pytest.raises(ValueError, match="calibration"):
        fitted.evaluate(rows, [], asof=START + timedelta(minutes=194))
    with pytest.raises(ValueError, match="annotation"):
        fitted.calibrate(rows, [], asof=START + timedelta(minutes=194))


@pytest.mark.parametrize("change", ["future", "overlap", "future_label", "constant", "mismatch"])
def test_calibration_delays_disjointness_and_labels_fail_closed(fitted, change):
    rows, labels = patterns(129, 64, unusual=True)
    cutoff = START + timedelta(minutes=194)
    if change == "future":
        cutoff = START + timedelta(minutes=191)
    elif change == "overlap":
        rows[0] = replace(rows[0], pattern_id="0")
        labels[0] = replace(labels[0], pattern_id="0")
    elif change == "future_label":
        labels[0] = replace(labels[0], available_time=cutoff + timedelta(minutes=1))
    elif change == "constant":
        labels = [replace(a, suspicious=False) for a in labels]
    else:
        labels[0] = replace(labels[0], pattern_id="wrong")
    identity = fitted.model_sha256
    with pytest.raises(ValueError):
        fitted.calibrate(rows, labels, asof=cutoff)
    assert fitted.model_sha256 == identity


def test_prediction_cannot_use_later_model_and_evaluation_cannot_reuse_data(calibrated):
    early, labels = patterns(129, 64, unusual=True)
    with pytest.raises(ValueError, match="after decision"):
        calibrated.predict(early)
    with pytest.raises(ValueError, match="disjoint"):
        calibrated.evaluate(early, labels, asof=START + timedelta(minutes=307))
    later, annotations = patterns(226, 64, unusual=True)
    annotations[0] = replace(annotations[0], available_time=START + timedelta(days=5))
    with pytest.raises(ValueError, match="annotation clock"):
        calibrated.evaluate(later, annotations, asof=START + timedelta(minutes=307))


def test_exact_safe_json_restore_and_exclusive_write(calibrated, tmp_path):
    rows, labels = patterns(226, 64, unusual=True)
    path = tmp_path / "model.json"
    calibrated.save(path)
    restored = TradeAnomalyModel.load(path)
    assert restored.predict(rows) == calibrated.predict(rows)
    assert restored.evaluate(
        rows, labels, asof=START + timedelta(minutes=307)
    ) == calibrated.evaluate(rows, labels, asof=START + timedelta(minutes=307))
    with pytest.raises(FileExistsError):
        calibrated.save(path)
    snapshot = json.loads(path.read_text())
    snapshot["payload"]["trees"][0]["threshold"][0] += 1
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="identity"):
        TradeAnomalyModel.load(path)


@pytest.mark.parametrize(
    "change",
    ["cycle", "unreachable", "child_count", "honesty", "calibration", "infinite", "features"],
)
def test_self_rehashed_malformed_tree_artifacts_are_rejected(calibrated, change):
    snapshot = calibrated.snapshot()
    payload = snapshot["payload"]
    tree = payload["trees"][0]
    if change == "cycle":
        tree["left"][0] = 0
    elif change == "unreachable":
        tree["right"][0] = tree["left"][0]
    elif change == "child_count":
        tree["count"][0] -= 1
    elif change == "honesty":
        payload["insider_misconduct"] = "PROVEN"
    elif change == "calibration":
        payload["calibration"]["scale"] = 0
    elif change == "features":
        payload["feature_names"] = ["different"] * 6
    else:
        tree["threshold"][0] = float("inf")
        with pytest.raises(ValueError):
            TradeAnomalyModel.restore(snapshot)
        return
    snapshot["model_sha256"] = _hash(payload)
    with pytest.raises(ValueError):
        TradeAnomalyModel.restore(snapshot)


def test_frozen_fit_and_parameter_seal(fitted):
    rows, _ = patterns(129, 64)
    with pytest.raises(ValueError, match="frozen"):
        fitted.fit(rows, asof=START + timedelta(minutes=194))
    fitted.config["seed"] += 1
    with pytest.raises(ValueError, match="mutated"):
        fitted.predict(rows)


def test_synthetic_annotations_on_caller_data_propagate_and_survive_restore():
    def caller_rows(first, n):
        # Evidence-class plumbing probe, not an empirical dataset or claim.
        return [
            replace(row, source="caller_supplied_tape", synthetic=False)
            for row in patterns(first, n, unusual=True)[0]
        ]

    model = TradeAnomalyModel(trees=8, samples=32).fit(
        caller_rows(0, 64), asof=START + timedelta(minutes=64)
    )
    model.calibrate(
        caller_rows(65, 64), patterns(65, 64, unusual=True)[1], asof=START + timedelta(minutes=130)
    )
    later = caller_rows(131, 64)
    labels = patterns(131, 64, unusual=True)[1]
    for candidate in (model, TradeAnomalyModel.restore(model.snapshot())):
        assert all(score.synthetic for score in candidate.predict(later))
        result = candidate.evaluate(later, labels, asof=START + timedelta(minutes=196))
        assert result["synthetic"]
        assert not result["synthetic_feature_data"]
        assert (
            result["synthetic_calibration_annotations"]
            and result["synthetic_evaluation_annotations"]
        )
        assert not result["market_evidence"]
    snapshot = model.snapshot()
    snapshot["payload"]["synthetic"] = False
    snapshot["model_sha256"] = _hash(snapshot["payload"])
    with pytest.raises(ValueError, match="synthetic annotations"):
        TradeAnomalyModel.restore(snapshot)


def test_rehashed_deep_comb_rejected_at_actual_isolation_fit_depth():
    model = TradeAnomalyModel(trees=1, samples=16).fit(
        patterns(0, 32)[0], asof=START + timedelta(minutes=32)
    )
    snapshot = model.snapshot()
    tree = {
        "left": [-1] * 31,
        "right": [-1] * 31,
        "feature": [-2] * 31,
        "threshold": [-2.0] * 31,
        "count": [1] * 31,
    }
    for depth in range(15):
        node = 2 * depth
        tree["left"][node] = node + 1
        tree["right"][node] = node + 2
        tree["feature"][node] = 0
        tree["threshold"][node] = 0.0
        tree["count"][node] = 16 - depth
    snapshot["payload"]["trees"] = [tree]
    snapshot["model_sha256"] = _hash(snapshot["payload"])
    with pytest.raises(ValueError, match="depth budget"):
        TradeAnomalyModel.restore(snapshot)


def test_restore_sample_count_must_match_training_configuration(fitted):
    snapshot = fitted.snapshot()
    snapshot["payload"]["config"]["samples"] = 128
    snapshot["model_sha256"] = _hash(snapshot["payload"])
    with pytest.raises(ValueError, match="sample count"):
        TradeAnomalyModel.restore(snapshot)


@pytest.mark.parametrize(
    "change", ["naive", "future", "nan", "bool", "domain", "dimension", "source", "synthetic"]
)
def test_trade_pattern_validation(change):
    row = patterns(0, 1)[0][0]
    overrides = {
        "naive": {"event_time": datetime(2020, 1, 1)},
        "future": {"available_time": row.decision_time + timedelta(seconds=1)},
        "nan": {"features": (float("nan"), *row.features[1:])},
        "bool": {"features": (True, *row.features[1:])},
        "domain": {"features": (*row.features[:5], 2)},
        "dimension": {"features": row.features[:2]},
        "source": {"source": ""},
        "synthetic": {"synthetic": 1},
    }
    with pytest.raises(ValueError):
        replace(row, **overrides[change])


def test_mixed_sources_and_future_suffix_are_rejected(fitted):
    rows, _ = patterns(129, 64)
    rows[-1] = replace(rows[-1], synthetic=False)
    with pytest.raises(ValueError, match="mixed"):
        fitted.predict(rows)
    with pytest.raises(ValueError, match="cutoff"):
        TradeAnomalyModel().fit(patterns(0, 128)[0], asof=START + timedelta(minutes=125))
