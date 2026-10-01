"""SYNTHETIC FedAvg mathematics/causality/learning checks, not privacy evidence."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from scipy.special import expit

from quant_fund.models.federated_market import (
    BinaryExample,
    BinaryTask,
    ClientConfig,
    ClientUpdate,
    FederatedClient,
    FederatedModel,
    FederatedServer,
    _hash,
    compare_models,
    evaluate_binary,
    train_centralized,
    transport_to_raw,
)

BASE = datetime(2020, 1, 1, tzinfo=UTC)
EVIDENCE = "a" * 64


def _task():
    return BinaryTask(
        "synthetic_binary_event",
        ("x1", "x2"),
        "positive_event",
        "label1 means the supplied next synthetic event is positive",
    )


def _row(task, i, features=(0.2, -0.4), label=1, *, prefix="a", start=0, split="train"):
    event = BASE + timedelta(hours=i + start)
    return BinaryExample(
        f"{prefix}:{i}",
        "ASSET",
        task.task_sha256,
        task.feature_names,
        tuple(float(v) for v in features),
        int(label),
        event,
        event + timedelta(minutes=1),
        event + timedelta(minutes=2),
        event + timedelta(minutes=5),
        event + timedelta(minutes=8),
        "synthetic_fixture",
        EVIDENCE,
        split,
        True,
    )


def _data(
    task, n=32, *, prefix="a", seed=11, mean=(0.0, 0.0), scale=(1.0, 1.0), start=0, split="train"
):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 2)) * scale + mean
    p = expit(features @ np.asarray([1.5, -0.8]) + 0.2)
    labels = rng.binomial(1, p)
    return tuple(
        _row(task, i, x, y, prefix=prefix, start=start, split=split)
        for i, (x, y) in enumerate(zip(features, labels, strict=True))
    )


def _clients(task):
    a = _data(task, 64, prefix="a", seed=44, mean=(-1.0, 0.0), scale=(0.7, 1.5))
    b = _data(task, 96, prefix="b", seed=82, mean=(1.0, 0.5), scale=(1.2, 0.3))
    return (FederatedClient("alpha", task, a), FederatedClient("beta", task, b)), a + b


def _round(server, clients, config=None):
    round_id = server.freeze().round_id + 1
    as_of = BASE + timedelta(hours=100, minutes=round_id)
    spec = server.begin_round(
        tuple(client.client_id for client in clients),
        training_as_of=BASE + timedelta(hours=100),
        aggregation_as_of=as_of,
        config=config,
    )
    base = server.freeze()
    results = tuple(client.train(spec, base, completed_at=as_of) for client in clients)
    receipt = server.aggregate(tuple(result.update for result in results), as_of=as_of)
    return spec, results, receipt


def _pending():
    task = _task()
    server = FederatedServer(task, initial_as_of=BASE - timedelta(hours=1))
    clients, _ = _clients(task)
    spec = server.begin_round(
        ("alpha", "beta"),
        training_as_of=BASE + timedelta(hours=100),
        aggregation_as_of=BASE + timedelta(hours=101),
    )
    results = tuple(
        client.train(spec, server.freeze(), completed_at=spec.aggregation_as_of)
        for client in clients
    )
    return server, spec, results


def test_parameter_transport_independently_preserves_logits_and_probabilities():
    weights = np.asarray([0.7, -1.2])
    bias = 0.3
    mean, scale = np.asarray([12.0, -8.0]), np.asarray([2.0, 0.25])
    raw, raw_bias = transport_to_raw(weights, bias, mean, scale)
    x = np.asarray([[10.0, -7.0], [15.0, -9.0], [100.0, 3.0]])
    local = ((x - mean) / scale) @ weights + bias
    original = x @ raw + raw_bias
    np.testing.assert_allclose(local, original, rtol=0, atol=1e-12)
    np.testing.assert_allclose(expit(local), expit(original), rtol=0, atol=1e-14)
    # Returning the global raw model to local coordinates is the inverse.
    np.testing.assert_allclose(raw * scale, weights)
    assert raw_bias + raw @ mean == pytest.approx(bias)


def test_prediction_is_typed_research_only_and_independent_of_unpublished_label():
    task = _task()
    server = FederatedServer(task, initial_as_of=BASE - timedelta(hours=1))
    row = _row(task, 0, label=0)
    one = server.freeze().predict(row)
    two = server.freeze().predict(
        replace(
            row, label=1, target_available_time=row.target_available_time + timedelta(hours=100)
        )
    )
    assert one == two
    assert one.probability == 0.5 and one.research_only
    assert not one.market_evidence and not one.privacy_guarantee
    assert len(one.prediction_sha256) == 64


def test_one_client_gradient_step_matches_independent_hand_logistic_gradient():
    task = _task()
    rows = tuple(
        _row(task, i, x, y)
        for i, (x, y) in enumerate(
            [((1.0, 2.0), 1), ((-1.0, 0.0), 0), ((2.0, -1.0), 1), ((0.0, 1.0), 0)]
        )
    )
    server = FederatedServer(task, initial_as_of=BASE - timedelta(hours=1))
    config = ClientConfig(epochs=1, batch_size=4, learning_rate=0.2, l2=0)
    spec = server.begin_round(
        ("client",),
        training_as_of=BASE + timedelta(hours=5),
        aggregation_as_of=BASE + timedelta(hours=6),
        config=config,
    )
    client = FederatedClient("client", task, rows)
    result = client.train(spec, server.freeze(), completed_at=spec.aggregation_as_of)
    x = np.asarray([row.features for row in rows])
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    error = np.asarray([0.5 - row.label for row in rows])
    expected = -0.2 * np.mean(z * error[:, None], axis=0)
    np.testing.assert_allclose(result.standardized_coefficients, expected, atol=1e-14)
    assert result.standardized_intercept == pytest.approx(-0.2 * error.mean())
    assert result.update.optimizer_steps == 1


def test_sample_weighted_fedavg_hand_calculation_and_frozen_client_order():
    task = _task()
    server = FederatedServer(task, initial_as_of=BASE - timedelta(hours=1))
    spec = server.begin_round(
        ("a", "b"),
        training_as_of=BASE + timedelta(hours=10),
        aggregation_as_of=BASE + timedelta(hours=11),
    )
    common = dict(
        round_id=1,
        round_sha256=spec.round_sha256,
        task_sha256=task.task_sha256,
        feature_names=task.feature_names,
        base_model_sha256=server.freeze().model_sha256,
        config_sha256=spec.client_config.config_sha256,
        optimizer_steps=5,
        training_cutoff=spec.training_as_of,
        max_feature_availability=BASE,
        max_target_availability=BASE,
        completed_at=spec.aggregation_as_of,
        training_source_sha256=EVIDENCE,
        training_synthetic=True,
    )
    one = ClientUpdate(
        client_id="a", coefficients=(1.0, 2.0), intercept=-1.0, n_examples=4, **common
    )
    two = ClientUpdate(
        client_id="b", coefficients=(4.0, -1.0), intercept=2.0, n_examples=8, **common
    )
    receipt = server.aggregate((two, one), as_of=spec.aggregation_as_of)
    assert receipt.sample_counts == (4, 8) and receipt.total_examples == 12
    assert receipt.sample_weights == (1 / 3, 2 / 3)
    np.testing.assert_allclose(receipt.model.coefficients, [3.0, 0.0], atol=1e-14)
    assert receipt.model.intercept == pytest.approx(1.0)
    assert tuple(update.client_id for update in receipt.updates) == ("a", "b")
    assert not receipt.privacy_guarantee and not receipt.market_evidence
    assert not receipt.client_metadata_independently_verified


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "missing",
        "foreign_client",
        "stale_round",
        "base",
        "task",
        "feature_order",
        "config",
        "future_completion",
        "budget",
        "deadline",
        "malformed",
        "mutated",
    ],
)
def test_entire_round_rejection_leaves_server_and_pending_state_unchanged(fault):
    server, spec, results = _pending()
    one, two = (result.update for result in results)
    values = [one, two]
    as_of = spec.aggregation_as_of
    if fault == "duplicate":
        values = [one, one]
    elif fault == "missing":
        values = [one]
    elif fault == "foreign_client":
        values = [replace(one, client_id="foreign"), two]
    elif fault == "stale_round":
        values = [replace(one, round_id=2), two]
    elif fault == "base":
        values = [replace(one, base_model_sha256="b" * 64), two]
    elif fault == "task":
        values = [replace(one, task_sha256="b" * 64), two]
    elif fault == "feature_order":
        values = [replace(one, feature_names=("x2", "x1")), two]
    elif fault == "config":
        values = [replace(one, config_sha256="b" * 64), two]
    elif fault == "future_completion":
        values = [replace(one, completed_at=as_of + timedelta(seconds=1)), two]
    elif fault == "budget":
        values = [replace(one, optimizer_steps=one.optimizer_steps + 1), two]
    elif fault == "deadline":
        as_of += timedelta(seconds=1)
    elif fault == "malformed":
        values = [{}, two]
    elif fault == "mutated":
        object.__setattr__(one, "intercept", 100.0)
    original = server.snapshot_payload()
    with pytest.raises(ValueError):
        server.aggregate(values, as_of=as_of)
    assert server.snapshot_payload() == original


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coefficients": (float("nan"), 0.0)},
        {"intercept": float("inf")},
        {"n_examples": 20_001},
        {"n_examples": True},
        {"optimizer_steps": 20_001},
        {"feature_names": ("x1",)},
        {"training_source_sha256": "bad"},
    ],
)
def test_malformed_or_unbounded_client_update_construction_is_rejected(kwargs):
    server, _, results = _pending()
    original = server.snapshot_payload()
    with pytest.raises(ValueError):
        replace(results[0].update, **kwargs)
    assert server.snapshot_payload() == original


def test_future_features_labels_or_naive_original_clocks_rejected():
    task = _task()
    row = _row(task, 0)
    with pytest.raises(ValueError, match="clocks"):
        replace(row, feature_available_time=row.decision_time + timedelta(seconds=1))
    with pytest.raises(ValueError, match="clocks"):
        replace(row, target_available_time=row.target_event_time - timedelta(seconds=1))
    with pytest.raises(ValueError, match="timezone"):
        replace(row, event_time=row.event_time.replace(tzinfo=None))


def test_train_prefix_scaling_and_parameters_are_independent_of_future_and_holdout_suffix():
    task = _task()
    training = _data(task, 16)
    suffix = _data(task, 6, start=200, prefix="future", seed=19)
    holdout = _data(task, 6, start=300, prefix="holdout", split="holdout", seed=31)
    original = FederatedClient("alpha", task, training + suffix + holdout)
    changed = FederatedClient(
        "alpha",
        task,
        training
        + tuple(
            replace(row, features=(1000.0, -2000.0), label=1 - row.label)
            for row in suffix + holdout
        ),
    )
    server = FederatedServer(task, initial_as_of=BASE - timedelta(hours=1))
    spec = server.begin_round(
        ("alpha",),
        training_as_of=BASE + timedelta(hours=100),
        aggregation_as_of=BASE + timedelta(hours=101),
    )
    first = original.train(spec, server.freeze(), completed_at=spec.aggregation_as_of)
    second = changed.train(spec, server.freeze(), completed_at=spec.aggregation_as_of)
    assert first.update == second.update and first.model == second.model
    assert first.feature_mean == second.feature_mean and first.feature_scale == second.feature_scale
    assert first.excluded_unpublished_rows == 6
    x = np.asarray([row.features for row in training])
    np.testing.assert_allclose(first.feature_mean, x.mean(axis=0))
    np.testing.assert_allclose(first.feature_scale, x.std(axis=0))
    assert set(first.training_ids) == {row.record_id for row in training}


def test_unpublished_client_labels_cannot_produce_update_and_cache_is_deterministic():
    task = _task()
    future = FederatedClient("alpha", task, _data(task, 4, start=200))
    server = FederatedServer(task, initial_as_of=BASE - timedelta(hours=1))
    spec = server.begin_round(
        ("alpha",),
        training_as_of=BASE + timedelta(hours=100),
        aggregation_as_of=BASE + timedelta(hours=101),
    )
    with pytest.raises(ValueError):
        future.train(spec, server.freeze(), completed_at=spec.aggregation_as_of)
    assert future.train_histories == ()
    client = FederatedClient("alpha", task, _data(task, 16))
    result = client.train(spec, server.freeze(), completed_at=spec.aggregation_as_of)
    assert result == client.train(spec, server.freeze(), completed_at=spec.aggregation_as_of)
    assert len(client.train_histories) == 1
    with pytest.raises(ValueError, match="cached"):
        client.train(
            spec, server.freeze(), completed_at=spec.aggregation_as_of - timedelta(minutes=1)
        )


def test_actual_local_gradients_and_federated_learning_on_independent_later_holdout():
    task = _task()
    clients, pooled = _clients(task)
    server = FederatedServer(task, initial_as_of=BASE - timedelta(hours=1))
    initial = server.freeze()
    for _ in range(6):
        _, results, _ = _round(server, clients)
    holdout = _data(task, 1024, prefix="evaluation", seed=131, start=2000, split="holdout")
    known = tuple(row.record_id for row in pooled)
    as_of = BASE + timedelta(hours=4000)
    before = evaluate_binary(initial, holdout, as_of=as_of, forbidden_training_ids=known)
    after = evaluate_binary(server.freeze(), holdout, as_of=as_of, forbidden_training_ids=known)
    assert after["brier"] < before["brier"] and after["log_loss"] < before["log_loss"]
    assert any(result.epoch_log_losses[-1] < result.epoch_log_losses[0] for result in results)
    assert after["synthetic"] and not after["market_evidence"] and not after["privacy_guarantee"]
    # Actual local standardized and transmitted raw predictors are equal.
    for result in results:
        x = np.asarray([row.features for row in holdout[:8]])
        local = (
            (x - result.feature_mean) / result.feature_scale
        ) @ result.standardized_coefficients + result.standardized_intercept
        raw = x @ result.update.coefficients + result.update.intercept
        np.testing.assert_allclose(local, raw, atol=1e-12)


def test_independent_brier_and_binary_log_score_arithmetic():
    task = _task()
    model = FederatedModel(task, (math.log(3), 0.0), 0.0, BASE, BASE, 0, 0, EVIDENCE, True)
    holdout = tuple(
        _row(task, i, (1.0 if i % 2 else -1.0, 0.0), i % 2, start=10, split="holdout")
        for i in range(4)
    )
    report = evaluate_binary(
        model, holdout, as_of=BASE + timedelta(hours=20), forbidden_training_ids=()
    )
    assert report["brier"] == pytest.approx(0.25**2)
    assert report["log_loss"] == pytest.approx(-math.log(0.75))


@pytest.mark.parametrize(
    "fault", ["reused_id", "train_split", "future_label", "model_not_available", "wrong_task"]
)
def test_holdout_evaluation_rejects_leakage_or_future_publication(fault):
    server, _, results = _pending()
    model = results[0].model
    task = model.task
    rows = list(_data(task, 4, prefix="holdout", start=200, split="holdout"))
    forbidden = ()
    if fault == "reused_id":
        forbidden = (rows[0].record_id,)
    elif fault == "train_split":
        rows[0] = replace(rows[0], split="train")
    elif fault == "future_label":
        rows[0] = replace(rows[0], target_available_time=BASE + timedelta(hours=2000))
    elif fault == "model_not_available":
        model = replace(model, available_as_of=BASE + timedelta(hours=1000))
    elif fault == "wrong_task":
        rows[0] = replace(rows[0], task_sha256="b" * 64)
    original = server.snapshot_payload()
    with pytest.raises(ValueError):
        evaluate_binary(
            model, rows, as_of=BASE + timedelta(hours=500), forbidden_training_ids=forbidden
        )
    assert server.snapshot_payload() == original


def test_global_local_centralized_control_retains_optimization_budget_and_privacy_limits():
    task = _task()
    clients, pooled = _clients(task)
    server = FederatedServer(task, initial_as_of=BASE - timedelta(hours=1))
    initial = server.freeze()
    for _ in range(3):
        _, local, _ = _round(server, clients)
    central = train_centralized(
        task,
        pooled,
        initial,
        config=ClientConfig(epochs=12, batch_size=32),
        training_as_of=BASE + timedelta(hours=100),
        completed_at=BASE + timedelta(hours=101),
    )
    holdout = _data(task, 64, prefix="evaluation", seed=912, start=200, split="holdout")
    report = compare_models(
        server.freeze(), local, central, holdout, as_of=BASE + timedelta(hours=500)
    )
    assert set(report["scores"]) == {"global", "centralized", "local:alpha", "local:beta"}
    assert (
        report["budgets"]["global_cumulative_optimizer_steps"]
        == server.freeze().cumulative_optimizer_steps
    )
    assert report["budgets"]["centralized_optimizer_steps"] == 12 * math.ceil(160 / 32)
    assert (
        not report["compute_budgets_equal"]
        and report["raw_training_rows_pooled_for_centralized_control"]
    )
    assert not report["privacy_guarantee"] and not report["process_isolation"]
    assert report["comparison_coordinator_sees_training_ids_and_holdout_rows"]
    assert all(
        not score["economic_independence_established"] for score in report["scores"].values()
    )


def test_json_restore_replays_aggregation_and_pending_round_exactly(tmp_path):
    task = _task()
    clients, _ = _clients(task)
    server = FederatedServer(task, initial_as_of=BASE - timedelta(hours=1))
    _round(server, clients)
    pending = server.begin_round(
        ("alpha", "beta"),
        training_as_of=BASE + timedelta(hours=100),
        aggregation_as_of=BASE + timedelta(hours=102),
    )
    path = tmp_path / "server.json"
    server.save_json(path)
    loaded = FederatedServer.load_json(path)
    assert loaded.snapshot_payload() == server.snapshot_payload()
    base = server.freeze()
    results = tuple(
        client.train(pending, base, completed_at=pending.aggregation_as_of) for client in clients
    )
    updates = tuple(result.update for result in results)
    one = server.aggregate(updates, as_of=pending.aggregation_as_of)
    two = loaded.aggregate(updates, as_of=pending.aggregation_as_of)
    assert one == two and server.snapshot_payload() == loaded.snapshot_payload()
    with pytest.raises(FileExistsError):
        server.save_json(path)


@pytest.mark.parametrize(
    "fault", ["hash", "model_coefficient", "count", "version", "runtime", "claim_flag"]
)
def test_json_tampering_rejected_even_with_rehashed_outer_payload(tmp_path, fault):
    task = _task()
    clients, _ = _clients(task)
    server = FederatedServer(task, initial_as_of=BASE - timedelta(hours=1))
    _round(server, clients)
    path = tmp_path / "snapshot.json"
    server.save_json(path)
    payload = json.loads(path.read_text())
    if fault == "hash":
        payload["model"]["intercept"] += 1
    elif fault == "model_coefficient":
        payload["receipts"][0]["model"]["coefficients"][0] += 1
    elif fault == "count":
        payload["receipts"][0]["sample_counts"][0] += 1
    elif fault == "version":
        payload["schema_version"] = "different"
    elif fault == "runtime":
        payload["numpy_version"] = "different"
    elif fault == "claim_flag":
        payload["model"]["privacy_guarantee"] = True
    if fault != "hash":
        payload.pop("snapshot_sha256")
        payload["snapshot_sha256"] = _hash(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        FederatedServer.load_json(path)


def test_aggregator_payload_contains_no_raw_rows_labels_or_local_normalizers():
    server, _, results = _pending()
    update_payloads = [asdict(result.update) for result in results]
    forbidden = {
        "examples",
        "rows",
        "features",
        "labels",
        "feature_mean",
        "feature_scale",
        "standardized_coefficients",
        "training_ids",
    }
    assert all(not (set(payload) & forbidden) for payload in update_payloads)
    receipt = server.aggregate(
        tuple(result.update for result in results), as_of=results[0].update.completed_at
    )
    assert receipt.sample_counts == (64, 96)
    assert not receipt.privacy_guarantee
    assert not server.freeze().privacy_guarantee


def test_deterministic_client_histories_and_models_with_same_protocol():
    task = _task()
    one, two = (
        FederatedServer(task, initial_as_of=BASE - timedelta(hours=1)),
        FederatedServer(task, initial_as_of=BASE - timedelta(hours=1)),
    )
    first, _ = _clients(task)
    second, _ = _clients(task)
    for _ in range(2):
        _, a, ra = _round(one, first)
        _, b, rb = _round(two, second)
        assert a == b and ra == rb
    assert one.snapshot_payload() == two.snapshot_payload()
    assert first[0].train_histories == second[0].train_histories
