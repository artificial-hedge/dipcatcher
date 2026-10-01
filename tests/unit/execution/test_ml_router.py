"""SYNTHETIC fitted-model, chronology and optimizer checks; no order submission."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from itertools import product

import numpy as np
import pytest
from scipy.special import expit

from quant_fund.execution.ml_router import (
    ChildOrderProposal,
    FillFeatures,
    MLFillRouter,
    OrderObservation,
)
from quant_fund.metrics.scoring import crps_gaussian, log_score_gaussian

START = datetime(2020, 1, 1, tzinfo=UTC)


def orders(n=500, offset=0, seed=7):
    rng = np.random.default_rng(seed)
    result = []
    for i in range(n):
        distance = float(rng.uniform(-3, 3))
        venue = ("A", "B", "C")[i % 3]
        probability = float(expit(-2 * distance + (0.4 if venue == "A" else -0.3)))
        filled = bool(rng.random() < probability)
        features = FillFeatures(distance, 2, 100, 500, 10, 30)
        clock = START + timedelta(minutes=offset + i)
        result.append(
            OrderObservation(
                str(offset + i),
                venue,
                clock,
                clock,
                clock + timedelta(seconds=30),
                features,
                filled,
                float(0.8 * distance + rng.normal(scale=0.3)) if filled else None,
                "SYNTHETIC_router_fixture",
                True,
            )
        )
    return result


def fitted():
    rows = orders()
    cutoff = rows[-1].outcome_available_time
    model = MLFillRouter(seed=7)
    model.update_models(rows, asof=cutoff)
    return model, rows, cutoff


def proposal(venue, clock, capacity=3, fee=0, price=100, distance=-1):
    return ChildOrderProposal(
        venue, clock, clock, price, capacity, fee, 1, FillFeatures(distance, 2, 100, 500, 10, 30)
    )


def test_fitted_probabilities_learn_and_out_of_time_proper_scores_match_reference():
    model, training, cutoff = fitted()
    test = orders(200, offset=1000, seed=19)
    metrics = model.evaluate(test, asof=test[-1].outcome_available_time)
    probabilities = np.asarray(
        [
            model.predict_fill(
                proposal(row.venue, row.decision_time, distance=row.features.price_distance_bps),
                decision_time=row.decision_time,
            ).p_fill
            for row in test
        ]
    )
    outcomes = np.asarray([int(row.filled) for row in test])
    baseline = np.mean([row.filled for row in training])
    assert metrics["fill_brier_score"] == pytest.approx(np.mean((probabilities - outcomes) ** 2))
    assert metrics["fill_brier_score"] < np.mean((baseline - outcomes) ** 2) - 0.05
    filled = [row for row in test if row.filled]
    means = np.asarray(
        [
            model.predict_fill(
                proposal(row.venue, row.decision_time, distance=row.features.price_distance_bps),
                decision_time=row.decision_time,
            ).toxicity_mean_bps
            for row in filled
        ]
    )
    targets = np.asarray([row.toxicity_bps for row in filled], dtype=float)
    scales = np.full(targets.shape, model.toxicity_scale)
    assert metrics["toxicity_crps"] == pytest.approx(crps_gaussian(targets, means, scales).mean())
    assert metrics["toxicity_log_score"] == pytest.approx(
        log_score_gaussian(targets, means, scales).mean()
    )
    assert metrics["synthetic"] and metrics["research_only"] and not metrics["market_evidence"]


def test_unpublished_outcomes_and_future_suffix_cannot_change_training():
    model, training, cutoff = fitted()
    second = MLFillRouter(seed=7)
    second.update_models(training + orders(50, offset=1000, seed=31), asof=cutoff)
    assert second.model_sha256 == model.model_sha256
    assert second.training_sha256 == model.training_sha256
    delayed = [replace(row, outcome_available_time=cutoff + timedelta(days=1)) for row in training]
    with pytest.raises(ValueError, match="16 finalized"):
        MLFillRouter().update_models(delayed, asof=cutoff)


def test_capacity_allocation_matches_exhaustive_linear_optimum():
    model, _rows, cutoff = fitted()
    clock = cutoff + timedelta(minutes=1)
    candidates = [
        proposal("A", clock, 2, fee=2),
        proposal("B", clock, 3, fee=0),
        proposal("C", clock, 2, fee=-1),
    ]
    plan = model.route_order(
        candidates, quantity=5, side="buy", decision_time=clock, opportunity_cost_bps=10
    )
    costs = []
    for candidate in candidates:
        estimate = model.predict_fill(candidate, decision_time=clock)
        costs.append(
            estimate.p_fill
            * (candidate.fee_bps + candidate.crossing_cost_bps + estimate.toxicity_mean_bps)
            + (1 - estimate.p_fill) * 10
        )
    feasible = [
        sum(q * c for q, c in zip(quantities, costs, strict=True)) / 5
        for quantities in product(range(3), range(4), range(3))
        if sum(quantities) == 5
    ]
    assert plan.mean_expected_cost_bps == pytest.approx(min(feasible))
    assert sum(child.quantity for child in plan.children) == 5
    assert all(
        child.quantity <= next(p.capacity for p in candidates if p.venue == child.venue)
        for child in plan.children
    )
    assert plan.synthetic_training and not plan.live_pnl_claim


def test_pit_feature_quotes_training_cutoff_and_limits_fail_closed():
    model, _rows, cutoff = fitted()
    clock = cutoff + timedelta(minutes=1)
    candidate = proposal("A", clock, capacity=10)
    for poisoned in (
        replace(candidate, quote_available_time=clock + timedelta(seconds=1)),
        replace(candidate, feature_available_time=clock + timedelta(seconds=1)),
    ):
        with pytest.raises(ValueError, match="not available"):
            model.route_order(
                [poisoned], quantity=1, side="buy", decision_time=clock, opportunity_cost_bps=10
            )
    with pytest.raises(ValueError, match="training cutoff"):
        model.route_order(
            [candidate],
            quantity=1,
            side="buy",
            decision_time=cutoff - timedelta(seconds=1),
            opportunity_cost_bps=10,
        )
    with pytest.raises(ValueError, match="capacity"):
        model.route_order(
            [candidate],
            quantity=1,
            side="buy",
            decision_time=clock,
            opportunity_cost_bps=10,
            limit_price=99,
        )
    with pytest.raises(ValueError, match="capacity"):
        model.route_order(
            [candidate],
            quantity=1,
            side="sell",
            decision_time=clock,
            opportunity_cost_bps=10,
            limit_price=101,
        )


def test_overlapping_unfinalized_mixed_or_unknown_evidence_rejected():
    model, training, cutoff = fitted()
    with pytest.raises(ValueError, match="unseen, strictly post-training"):
        model.evaluate(training[:2], asof=cutoff)
    test = orders(20, offset=1000, seed=19)
    with pytest.raises(ValueError, match="finalized"):
        model.evaluate(test, asof=cutoff)
    with pytest.raises(ValueError, match="source"):
        model.evaluate(
            [replace(row, source="different") for row in test], asof=test[-1].outcome_available_time
        )
    with pytest.raises(ValueError, match="unseen venue"):
        model.predict_fill(proposal("unknown", cutoff), decision_time=cutoff)
    with pytest.raises(ValueError, match="duplicate order"):
        MLFillRouter().update_models(training + [training[0]], asof=cutoff)
    with pytest.raises(ValueError, match="mix data"):
        MLFillRouter().update_models(
            [replace(training[0], source="empirical", synthetic=False), *training[1:]], asof=cutoff
        )


def test_typed_feature_and_outcome_contracts():
    row = orders(1)[0]
    with pytest.raises(ValueError, match="features must be available"):
        replace(row, feature_available_time=row.decision_time + timedelta(seconds=1))
    with pytest.raises(ValueError, match="strictly after"):
        replace(row, outcome_available_time=row.decision_time)
    with pytest.raises(ValueError, match="synthetic source"):
        replace(row, synthetic=False)
    with pytest.raises(ValueError, match="numeric bounds"):
        FillFeatures(0, float("nan"), 1, 1, 1, 1)
    with pytest.raises(ValueError, match="numeric bounds"):
        FillFeatures(0, 1, -1, 1, 1, 1)
    with pytest.raises(ValueError, match="not fitted"):
        MLFillRouter().predict_fill(proposal("A", START), decision_time=START)
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(row, decision_time=datetime(2020, 1, 1))


def test_direct_prediction_cannot_bypass_training_or_quote_clock():
    model, _rows, cutoff = fitted()
    with pytest.raises(ValueError, match="training cutoff"):
        model.predict_fill(proposal("A", START), decision_time=START)
    with pytest.raises(ValueError, match="not available"):
        model.predict_fill(proposal("A", cutoff + timedelta(seconds=1)), decision_time=cutoff)


@pytest.mark.parametrize("field", ["fill", "scaler", "toxicity", "source", "synthetic"])
def test_model_identity_detects_mutated_parameters_and_evidence(field):
    model, _rows, cutoff = fitted()
    if field == "fill":
        model.fill.intercept_[0] += 10
    elif field == "scaler":
        model.scaler.mean_[0] += 10
    elif field == "toxicity":
        model.toxicity.coef_[0] += 10
    elif field == "source":
        model.source = "replacement"
    else:
        model.synthetic_training = False
    with pytest.raises(ValueError, match="parameters or provenance changed"):
        model.predict_fill(proposal("A", cutoff), decision_time=cutoff)
