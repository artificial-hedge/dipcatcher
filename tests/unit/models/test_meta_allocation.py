"""SYNTHETIC MAML correctness/gradient/constraint tests, not market evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant_fund.models.meta_allocation import (
    AdaptedGaussianModel,
    AllocationConstraints,
    CovarianceEvidence,
    FeaturePoint,
    GaussianPrediction,
    LabelledPoint,
    MAMLConfig,
    MAMLRegimeLearner,
    ParameterState,
    RegimeTask,
    allocate_research,
)

START = datetime(2025, 1, 1, tzinfo=UTC)
SHA = hashlib.sha256(b"SYNTHETIC MAML generator").hexdigest()
HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="optional torch nn extra unavailable")


def point(point_id: str, decision: datetime, x: float, entity: str = "SYNTHETIC_A") -> FeaturePoint:
    return FeaturePoint(
        point_id,
        entity,
        decision,
        decision - timedelta(minutes=2),
        decision - timedelta(minutes=1),
        (x,),
        "SYNTHETIC_FEATURE_GENERATOR",
        SHA,
        True,
    )


def row(point_id: str, decision: datetime, x: float, y: float) -> LabelledPoint:
    return LabelledPoint(
        point(point_id, decision, x),
        y,
        decision + timedelta(hours=1),
        decision + timedelta(hours=1, minutes=1),
        "SYNTHETIC_LABEL_GENERATOR",
        SHA,
        True,
    )


def task(index: int, task_id: str | None = None) -> RegimeTask:
    start = START + timedelta(days=index)
    slope = (-0.6, 0.3, 0.8)[index % 3]
    offset = 0.03 * index
    support = tuple(
        row(
            f"t{index}s{j}",
            start + timedelta(hours=2 * j),
            x,
            slope * x + offset + 0.015 * math.sin(j),
        )
        for j, x in enumerate((-1.0, -0.3, 0.3, 1.0))
    )
    query = tuple(
        row(
            f"t{index}q{j}",
            start + timedelta(hours=10 + 2 * j),
            x,
            slope * x + offset + 0.015 * math.cos(j),
        )
        for j, x in enumerate((-0.8, 0.2, 0.9))
    )
    return RegimeTask(task_id or f"train{index}", f"regime{index}", support, query)


def config(**kwargs: Any) -> MAMLConfig:
    return MAMLConfig(
        input_dim=1,
        hidden_dim=4,
        inner_steps=1,
        outer_steps=30,
        horizon=timedelta(hours=1),
        embargo=timedelta(minutes=5),
        **kwargs,
    )


@pytest.fixture(scope="module", autouse=True)
def cpu_threads() -> Iterator[None]:
    if not HAS_TORCH:
        yield
        return
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def fitted() -> MAMLRegimeLearner:
    if not HAS_TORCH:
        pytest.skip("optional torch nn extra unavailable")
    return MAMLRegimeLearner(config()).fit(
        (task(0), task(1), task(2)),
        fit_cutoff=START + timedelta(days=3),
        heldout_task_ids=("held4", "held5"),
    )


def test_numeric_clock_source_synthetic_and_immutable_schemas() -> None:
    p = point("p", START, 1.0)
    for kwargs in (
        {"values": (float("nan"),)},
        {"synthetic": 1},
        {"source_sha256": "bad"},
        {"values": [1.0]},
        {"decision_time": START.replace(tzinfo=None)},
        {"available_time": START + timedelta(seconds=1)},
    ):
        with pytest.raises(ValueError):
            replace(p, **kwargs)
    r = row("r", START, 1.0, 0.2)
    with pytest.raises(ValueError, match="target event"):
        replace(r, target_event_time=START)
    with pytest.raises(ValueError, match="label_synthetic"):
        replace(r, label_synthetic=1)
    with pytest.raises(ValueError, match="disjoint"):
        RegimeTask("reuse", "regime", (r,), (r,))
    with pytest.raises(ValueError, match="immutable"):
        RegimeTask("list", "regime", [r], (row("r2", START + timedelta(days=1), 1.0, 0.2),))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_dim": True},
        {"hidden_dim": 0},
        {"inner_steps": 21},
        {"outer_steps": 301},
        {"inner_lr": float("nan")},
        {"outer_lr": 0.0},
        {"seed": -1},
        {"horizon": timedelta(0)},
        {"embargo": timedelta(minutes=-1)},
        {"max_tasks": 65},
    ],
)
def test_config_resource_gates(kwargs: dict[str, Any]) -> None:
    defaults: dict[str, Any] = {"input_dim": 1}
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        MAMLConfig(**defaults)


def test_invalid_training_data_fails_before_optional_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    m = MAMLRegimeLearner(config())
    with pytest.raises(ValueError, match="unpublished"):
        m.fit((task(0),), fit_cutoff=START + timedelta(hours=3), heldout_task_ids=("held",))
    with pytest.raises(ValueError, match="disjoint"):
        m.fit((task(0), task(0)), fit_cutoff=START + timedelta(days=3), heldout_task_ids=("held",))
    with pytest.raises(ValueError, match="heldout"):
        m.fit((task(0),), fit_cutoff=START + timedelta(days=3), heldout_task_ids=("train0",))
    tiny = MAMLRegimeLearner(replace(config(), max_gradient_evaluations=1))
    with pytest.raises(ValueError, match="resource budget"):
        tiny.fit((task(0),), fit_cutoff=START + timedelta(days=3), heldout_task_ids=("held",))


def test_module_imports_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_fund.models.meta_allocation as module

    monkeypatch.setitem(sys.modules, "torch", None)
    spec = importlib.util.spec_from_file_location("_meta_no_torch", module.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_meta_no_torch", probe)
    spec.loader.exec_module(probe)
    assert probe.AllocationConstraints((0.5,)).budget == 1.0
    with pytest.raises(ImportError, match="optional nn extra"):
        MAMLRegimeLearner(config()).fit(
            (task(0),), fit_cutoff=START + timedelta(days=1), heldout_task_ids=("held",)
        )


@requires_torch
def test_second_order_meta_gradient_matches_full_inner_update_finite_difference() -> None:
    learner = MAMLRegimeLearner(config())
    tasks = (task(0),)
    cutoff = START + timedelta(days=1)
    audit = learner.meta_gradient(tasks, fit_cutoff=cutoff)
    assert audit.second_order and math.isfinite(audit.objective_nll)
    # Finite difference recomputes support SGD, so identity/FOMAML gradients
    # cannot pass by differentiating query loss alone.
    parameter_index, element = 2, 0  # w_mean
    epsilon = 1e-5

    def changed(delta: float) -> ParameterState:
        values = [list(v) for v in audit.parameters.values]
        values[parameter_index][element] += delta
        return ParameterState(audit.parameters.shapes, tuple(tuple(v) for v in values))

    plus = learner.meta_gradient(
        tasks, fit_cutoff=cutoff, parameters=changed(epsilon)
    ).objective_nll
    minus = learner.meta_gradient(
        tasks, fit_cutoff=cutoff, parameters=changed(-epsilon)
    ).objective_nll
    derivative = (plus - minus) / (2 * epsilon)
    assert audit.gradients.values[parameter_index][element] == pytest.approx(
        derivative, rel=2e-5, abs=1e-7
    )
    assert sum(abs(v) for values in audit.gradients.values for v in values) > 0.01


@requires_torch
def test_real_learning_frozen_initializations_and_rng_threads(
    fitted: MAMLRegimeLearner, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    receipt = fitted.metadata()
    assert receipt["second_order"] and receipt["data_label"] == "SYNTHETIC"
    assert (
        not receipt["market_evidence"] and not receipt["pretrained"] and not receipt["sota_claim"]
    )
    assert fitted.initialization("maml").sha256 != fitted.initialization("scratch").sha256
    assert fitted.initialization("pooled").sha256 != fitted.initialization("scratch").sha256
    assert receipt["meta_query_nll"][-1] < receipt["meta_query_nll"][0]
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()

    def forbidden_seed(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the caller's global RNG must not be seeded")

    monkeypatch.setattr(torch, "manual_seed", forbidden_seed)
    # Even a caller's non-CPU default must not change this module's CPU identity.
    with torch.device("meta"):
        m = MAMLRegimeLearner(replace(config(), outer_steps=2)).fit(
            (task(0),),
            fit_cutoff=START + timedelta(days=1),
            heldout_task_ids=("held",),
        )
    assert m.model_sha256 is not None
    assert torch.equal(rng, torch.get_rng_state()) and threads == torch.get_num_threads()
    original_hash = fitted.initialization().sha256
    copied = fitted.initialization().arrays()
    copied["w1"][:] = 1e6
    assert fitted.initialization().sha256 == original_hash
    with pytest.raises(AttributeError):
        fitted.config = config()  # type: ignore[misc]
    with pytest.raises(RuntimeError, match="frozen"):
        fitted.fit((task(0),), fit_cutoff=START + timedelta(days=1), heldout_task_ids=("held",))


@requires_torch
def test_task_separation_horizon_and_support_query_embargo() -> None:
    m = MAMLRegimeLearner(config())
    t = task(0)
    cutoff = START + timedelta(days=3)
    with pytest.raises(ValueError, match="regime/task IDs"):
        m.meta_gradient((t, replace(task(1), regime_id=t.regime_id)), fit_cutoff=cutoff)
    overlapping = replace(
        t,
        task_id="other",
        regime_id="other",
        support=tuple(
            replace(r, point=replace(r.point, point_id="copy" + r.point.point_id))
            for r in t.support
        ),
        query=tuple(
            replace(r, point=replace(r.point, point_id="copy" + r.point.point_id)) for r in t.query
        ),
    )
    with pytest.raises(ValueError, match="observation key"):
        m.meta_gradient((t, overlapping), fit_cutoff=cutoff)
    q = t.query[0]
    early_decision = START + timedelta(hours=7, minutes=2)
    early = replace(
        q,
        point=replace(
            q.point,
            decision_time=early_decision,
            event_time=early_decision - timedelta(minutes=2),
            available_time=early_decision - timedelta(minutes=1),
        ),
        target_event_time=early_decision + timedelta(hours=1),
        target_available_time=early_decision + timedelta(hours=1, minutes=1),
    )
    with pytest.raises(ValueError, match="purge/embargo"):
        m.meta_gradient((replace(t, query=(early, *t.query[1:])),), fit_cutoff=cutoff)
    with pytest.raises(ValueError, match="forecast horizon"):
        MAMLRegimeLearner(replace(config(), horizon=timedelta(hours=2))).meta_gradient(
            (t,), fit_cutoff=cutoff
        )


@requires_torch
def test_query_labels_never_adapt_support_or_preprocessing(fitted: MAMLRegimeLearner) -> None:
    t = task(4, "held4")
    asof = max(r.target_available_time for r in t.support)
    a = fitted.adapt(t.task_id, t.support, asof=asof)
    changed = replace(t, query=tuple(replace(r, target=100.0) for r in t.query))
    b = fitted.adapt(changed.task_id, changed.support, asof=asof)
    assert a.adapted_sha256 == b.adapted_sha256
    scores_a = a.score(t.query, asof=START + timedelta(days=5))
    scores_b = b.score(changed.query, asof=START + timedelta(days=5))
    assert scores_a != scores_b  # Query observations evaluate the frozen model only.
    x = np.array([r.point.values for r in t.support])
    np.testing.assert_allclose(a.preprocessing.x_mean, x.mean(axis=0))
    assert a.support_rows == len(t.support)


@requires_torch
def test_unpublished_support_and_future_prediction_suffix_invariance(
    fitted: MAMLRegimeLearner,
) -> None:
    t = task(4, "held4")
    asof = max(r.target_available_time for r in t.support)
    a = fitted.adapt(t.task_id, t.support, asof=asof)
    future = row("future-support", START + timedelta(days=7), 1000.0, -1000.0)
    b = fitted.adapt(t.task_id, t.support + (future,), asof=asof)
    assert a.adapted_sha256 == b.adapted_sha256
    prediction_cutoff = t.query[-1].point.decision_time
    baseline = a.predict(tuple(r.point for r in t.query), asof=prediction_cutoff)
    extended = a.predict(tuple(r.point for r in t.query) + (future.point,), asof=prediction_cutoff)
    assert baseline.forecast_sha256 == extended.forecast_sha256
    assert baseline.synthetic and not math.isnan(
        a.score(t.query, asof=START + timedelta(days=5))["crps"]
    )
    with pytest.raises(ValueError, match="published support labels"):
        fitted.adapt(t.task_id, (future,), asof=asof)


@requires_torch
def test_heldout_ids_decision_clocks_and_source_identity(fitted: MAMLRegimeLearner) -> None:
    t = task(4, "held4")
    asof = max(r.target_available_time for r in t.support)
    with pytest.raises(ValueError, match="declared held-out"):
        fitted.adapt("unregistered", t.support, asof=asof)
    with pytest.raises(ValueError, match="training point_id"):
        fitted.adapt("held4", task(0).support, asof=asof)
    past = tuple(
        replace(r, point=replace(r.point, point_id="past" + r.point.point_id))
        for r in task(0).support
    )
    with pytest.raises(ValueError, match="fit cutoff/embargo"):
        fitted.adapt("held4", past, asof=asof)
    adapted = fitted.adapt("held4", t.support, asof=asof)
    with pytest.raises(ValueError, match="adaptation cutoff/embargo"):
        adapted.predict(tuple(r.point for r in t.support), asof=asof)
    with pytest.raises(ValueError, match="unpublished"):
        adapted.score(t.query, asof=t.query[-1].point.decision_time)
    non_synthetic_support = tuple(
        replace(r, point=replace(r.point, synthetic=False), label_synthetic=False)
        for r in t.support
    )
    carried = fitted.adapt("held4", non_synthetic_support, asof=asof)
    assert carried.synthetic  # Synthetic meta-initialization remains explicit.


@requires_torch
def test_matched_support_adaptation_comparison_retains_all_outcomes(
    fitted: MAMLRegimeLearner,
) -> None:
    tasks = (task(4, "held4"), task(5, "held5"))
    result = fitted.compare_initializations(tasks, asof=START + timedelta(days=6))
    assert result["adaptation_budget_matched"] and result["support_matched"]
    assert not result["pretraining_compute_matched"] and not result["benefit_asserted"]
    assert not result["market_evidence"] and result["synthetic"]
    assert len(result["outcomes"]) == 2 * 3 * 2
    for task_id in ("held4", "held5"):
        for step in (0, 1):
            outcomes = [
                r for r in result["outcomes"] if r["task_id"] == task_id and r["steps"] == step
            ]
            assert {r["arm"] for r in outcomes} == {"maml", "pooled", "scratch"}
            assert {r["support_rows"] for r in outcomes} == {4}
            assert all(
                math.isfinite(r["gaussian_nll"]) and math.isfinite(r["crps"]) for r in outcomes
            )


@requires_torch
def test_frozen_json_persistence_and_hash_tampering(
    fitted: MAMLRegimeLearner, tmp_path: Path
) -> None:
    path = tmp_path / "model.json"
    fitted.save(path)
    restored = MAMLRegimeLearner.load(path)
    assert restored.model_sha256 == fitted.model_sha256
    assert restored.initialization().sha256 == fitted.initialization().sha256
    t = task(4, "held4")
    asof = max(r.target_available_time for r in t.support)
    assert (
        restored.adapt(t.task_id, t.support, asof=asof).adapted_sha256
        == fitted.adapt(t.task_id, t.support, asof=asof).adapted_sha256
    )
    payload = json.loads(path.read_text())
    payload["receipt"]["states"]["maml"]["values"][0][0] += 0.1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        MAMLRegimeLearner.load(path)


def forecast(kind: str = "forward_simple_return") -> GaussianPrediction:
    points = (point("a", START, 0.1, "SYNTHETIC_A"), point("b", START, 0.2, "SYNTHETIC_B"))
    return GaussianPrediction(
        points, (0.05, -0.03), (0.2, 0.3), kind, timedelta(hours=1), SHA, True
    )  # type: ignore[arg-type]


def test_long_only_budget_caps_risk_and_cash_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    f = forecast()
    constraints = AllocationConstraints((0.6, 0.4), budget=0.8, risk_aversion=2.0)
    allocation = allocate_research(f, constraints, asof=START)
    assert 0 <= allocation.weights[0] <= 0.6 and allocation.weights[1] == pytest.approx(
        0.0, abs=1e-10
    )
    assert sum(allocation.weights) <= 0.8 and sum(
        allocation.weights
    ) + allocation.cash == pytest.approx(0.8)
    assert "PROXY" in allocation.covariance_kind
    assert allocation.synthetic and allocation.research_only and not allocation.trading_policy
    assert not allocation.market_evidence
    restricted = allocate_research(f, replace(constraints, variance_limit=0.001), asof=START)
    assert restricted.predicted_variance <= 0.001 + 1e-8
    assert restricted.weights[0] < allocation.weights[0]
    cash = allocate_research(f, replace(constraints, variance_limit=0), asof=START)
    assert cash.weights == (0.0, 0.0) and cash.cash == 0.8


def test_dated_covariance_alignment_and_units_fail_closed() -> None:
    f = forecast()
    covariance = CovarianceEvidence(
        tuple(p.entity_id for p in f.points),
        ((0.04, 0.01), (0.01, 0.09)),
        f.horizon,
        START - timedelta(seconds=1),
        "SYNTHETIC_COVARIANCE",
        SHA,
        True,
    )
    constraints = AllocationConstraints((0.6, 0.4))
    a = allocate_research(f, constraints, asof=START, covariance=covariance)
    assert a.covariance_kind == "supplied_dated_covariance_evidence"
    assert a.covariance_sha256 != allocate_research(f, constraints, asof=START).covariance_sha256
    with pytest.raises(ValueError, match="unpublished"):
        allocate_research(
            f,
            constraints,
            asof=START,
            covariance=replace(covariance, available_time=START + timedelta(seconds=1)),
        )
    with pytest.raises(ValueError, match="entity/horizon"):
        allocate_research(
            f, constraints, asof=START, covariance=replace(covariance, horizon=timedelta(days=2))
        )
    with pytest.raises(ValueError, match="forward_simple_return"):
        allocate_research(forecast("numeric_outcome"), constraints, asof=START)
    with pytest.raises(ValueError, match="current decision"):
        allocate_research(f, constraints, asof=START + timedelta(seconds=1))
    with pytest.raises(ValueError, match="positive semidefinite"):
        replace(covariance, values=((1.0, 2.0), (2.0, 1.0)))
    with pytest.raises(ValueError, match="symmetric"):
        replace(covariance, values=((1.0, 0.2), (0.1, 1.0)))


@requires_torch
def test_actual_adapted_forecasts_feed_research_allocation() -> None:
    m = MAMLRegimeLearner(replace(config(), outer_steps=2, target_kind="forward_simple_return"))
    m.fit((task(1),), fit_cutoff=START + timedelta(days=3), heldout_task_ids=("held4",))
    t = task(4, "held4")
    adapted: AdaptedGaussianModel = m.adapt(
        "held4", t.support, asof=max(r.target_available_time for r in t.support)
    )
    decision = t.query[0].point.decision_time
    points = (
        replace(t.query[0].point, point_id="asset_a", entity_id="SYNTHETIC_A"),
        replace(t.query[0].point, point_id="asset_b", entity_id="SYNTHETIC_B", values=(0.7,)),
    )
    prediction = adapted.predict(points, asof=decision)
    allocation = allocate_research(
        prediction, AllocationConstraints((0.5, 0.5), variance_limit=0.02), asof=decision
    )
    assert sum(allocation.weights) <= 1 and all(0 <= w <= 0.5 for w in allocation.weights)
    assert allocation.predicted_variance <= 0.02 + 1e-8
    assert allocation.forecast_sha256 == prediction.forecast_sha256 and allocation.synthetic
