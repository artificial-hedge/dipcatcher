"""SYNTHETIC correctness fixtures for independently implemented proper scores.

These hand calculations and mathematical identities are not market evidence.
"""

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations import (
    score_binary_forecasts,
    score_empirical_crps,
    score_intervals,
    score_quantiles,
)
from fx1.operations.base import OperationContext


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def test_quantile_pinball_hand_calculation(context: OperationContext):
    request = score_quantiles.Input(
        outcomes=[1.0, 3.0],
        levels=[0.25, 0.5, 0.75],
        quantiles=[[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]],
    )
    result = score_quantiles.execute(request, context)
    assert result.observation_count == 2
    assert result.mean_pinball_by_level == pytest.approx([0.25, 0.0, 0.25])
    assert result.mean_pinball == pytest.approx(1 / 6)


def test_median_pinball_is_half_absolute_error(context: OperationContext):
    result = score_quantiles.execute(
        score_quantiles.Input(
            outcomes=[2.0, -3.0, 1.0], levels=[0.5], quantiles=[[0.0], [1.0], [1.0]]
        ),
        context,
    )
    assert result.mean_pinball == pytest.approx((2 + 4 + 0) / (2 * 3))


def test_quantile_ties_are_valid_for_discrete_distributions(context: OperationContext):
    result = score_quantiles.execute(
        score_quantiles.Input(outcomes=[2.0], levels=[0.1, 0.9], quantiles=[[2.0, 2.0]]),
        context,
    )
    assert result.mean_pinball == 0.0


@pytest.mark.parametrize(
    "arguments",
    [
        {"outcomes": [], "levels": [0.5], "quantiles": []},
        {"outcomes": [0.0], "levels": [0.5], "quantiles": [[0.0], [1.0]]},
        {"outcomes": [0.0], "levels": [0.25, 0.75], "quantiles": [[0.0]]},
        {"outcomes": [0.0], "levels": [0.75, 0.25], "quantiles": [[0.0, 1.0]]},
        {"outcomes": [0.0], "levels": [0.5, 0.5], "quantiles": [[0.0, 1.0]]},
        {"outcomes": [0.0], "levels": [0.25, 0.75], "quantiles": [[1.0, 0.0]]},
        {"outcomes": [0.0], "levels": [0.0], "quantiles": [[0.0]]},
        {"outcomes": [0.0], "levels": [1.0], "quantiles": [[0.0]]},
        {"outcomes": [True], "levels": [0.5], "quantiles": [[0.0]]},
        {"outcomes": [0.0], "levels": [0.5], "quantiles": [[float("nan")]]},
    ],
)
def test_quantile_invalid_inputs(arguments):
    with pytest.raises(ValidationError):
        score_quantiles.Input.model_validate(arguments)


def test_quantile_total_work_bound():
    with pytest.raises(ValidationError, match="250000"):
        score_quantiles.Input(
            outcomes=[0.0] * 2501,
            levels=[(index + 1) / 101 for index in range(100)],
            quantiles=[[0.0] * 100] * 2501,
        )


def test_binary_scores_hand_calculation(context: OperationContext):
    result = score_binary_forecasts.execute(
        score_binary_forecasts.Input(outcomes=[1, 0], probabilities=[0.8, 0.3]), context
    )
    assert result.brier_score == pytest.approx((0.04 + 0.09) / 2)
    assert result.mean_log_loss == pytest.approx(-math.log(0.8 * 0.7) / 2)
    assert result.log_loss_status == "finite"
    assert result.impossible_event_count == result.clipped_probability_count == 0


def test_correct_binary_endpoints_have_zero_loss(context: OperationContext):
    result = score_binary_forecasts.execute(
        score_binary_forecasts.Input(outcomes=[0, 1], probabilities=[0.0, 1.0]), context
    )
    assert result.brier_score == result.mean_log_loss == 0.0
    assert result.log_loss_status == "finite"


def test_impossible_binary_endpoints_are_explicit_infinite_loss(context: OperationContext):
    result = score_binary_forecasts.OPERATION.invoke(
        {"outcomes": [1, 0, 1], "probabilities": [0.0, 1.0, 0.5]}, context
    )
    score = result["result"]
    assert score["brier_score"] == pytest.approx(0.75)
    assert score["mean_log_loss"] is None
    assert score["log_loss_status"] == "positive_infinity"
    assert score["impossible_event_count"] == 2
    assert result["market_evidence"] is False


def test_binary_clipping_is_opt_in_and_preserves_original_brier(context: OperationContext):
    result = score_binary_forecasts.execute(
        score_binary_forecasts.Input(
            outcomes=[1, 0, 1], probabilities=[0.0, 1.0, 0.5], probability_clip=0.01
        ),
        context,
    )
    assert result.brier_score == pytest.approx(0.75)
    assert result.mean_log_loss == pytest.approx(-(2 * math.log(0.01) + math.log(0.5)) / 3)
    assert result.impossible_event_count == result.clipped_probability_count == 2
    assert result.log_loss_status == "finite"


def test_binary_complement_invariance(context: OperationContext):
    forward = score_binary_forecasts.execute(
        score_binary_forecasts.Input(outcomes=[1, 0, 1], probabilities=[0.2, 0.1, 0.6]),
        context,
    )
    complement = score_binary_forecasts.execute(
        score_binary_forecasts.Input(outcomes=[0, 1, 0], probabilities=[0.8, 0.9, 0.4]),
        context,
    )
    assert forward.brier_score == pytest.approx(complement.brier_score)
    assert forward.mean_log_loss == pytest.approx(complement.mean_log_loss)


@pytest.mark.parametrize(
    "arguments",
    [
        {"outcomes": [], "probabilities": []},
        {"outcomes": [1], "probabilities": [0.3, 0.5]},
        {"outcomes": [2], "probabilities": [0.3]},
        {"outcomes": [1.0], "probabilities": [0.3]},
        {"outcomes": [True], "probabilities": [0.3]},
        {"outcomes": [1], "probabilities": [True]},
        {"outcomes": [1], "probabilities": [-0.1]},
        {"outcomes": [1], "probabilities": [1.1]},
        {"outcomes": [1], "probabilities": [float("inf")]},
        {"outcomes": [1], "probabilities": [0.3], "probability_clip": 0.0},
        {"outcomes": [1], "probabilities": [0.3], "probability_clip": 0.5},
        {"outcomes": [1], "probabilities": [0.3], "probability_clip": True},
    ],
)
def test_binary_invalid_inputs(arguments):
    with pytest.raises(ValidationError):
        score_binary_forecasts.Input.model_validate(arguments)


def test_interval_score_hand_calculation(context: OperationContext):
    result = score_intervals.execute(
        score_intervals.Input(
            outcomes=[-1.0, 1.0, 4.0],
            lower=[0.0, 0.0, 0.0],
            upper=[2.0, 2.0, 2.0],
            nominal_coverage=0.8,
        ),
        context,
    )
    assert result.interval_score == pytest.approx((12 + 2 + 22) / 3)
    assert result.mean_width == 2.0
    assert result.mean_lower_miss_penalty == pytest.approx(10 / 3)
    assert result.mean_upper_miss_penalty == pytest.approx(20 / 3)
    assert result.empirical_coverage == pytest.approx(1 / 3)


def test_interval_endpoints_are_covered_without_penalty(context: OperationContext):
    result = score_intervals.execute(
        score_intervals.Input(
            outcomes=[0.0, 2.0, 1.0],
            lower=[0.0, 0.0, 1.0],
            upper=[2.0, 2.0, 1.0],
            nominal_coverage=0.9,
        ),
        context,
    )
    assert result.empirical_coverage == 1.0
    assert result.mean_lower_miss_penalty == result.mean_upper_miss_penalty == 0.0
    assert result.interval_score == pytest.approx(4 / 3)


def test_interval_equals_scaled_sum_of_tail_pinball_losses(context: OperationContext):
    outcomes = [-3.0, 0.0, 8.0]
    lower = [-1.0, -1.0, 2.0]
    upper = [1.0, 3.0, 4.0]
    alpha = 0.2
    result = score_intervals.execute(
        score_intervals.Input(
            outcomes=outcomes, lower=lower, upper=upper, nominal_coverage=1 - alpha
        ),
        context,
    )
    expected_scores = []
    for outcome, lo, hi in zip(outcomes, lower, upper, strict=True):
        lower_error, upper_error = outcome - lo, outcome - hi
        lower_loss = max(alpha / 2 * lower_error, (alpha / 2 - 1) * lower_error)
        upper_loss = max((1 - alpha / 2) * upper_error, -alpha / 2 * upper_error)
        expected_scores.append(2 / alpha * (lower_loss + upper_loss))
    assert result.interval_score == pytest.approx(sum(expected_scores) / len(outcomes))


@pytest.mark.parametrize(
    "replacement",
    [
        {"outcomes": []},
        {"outcomes": [0.0, 1.0]},
        {"lower": [2.0]},
        {"lower": [True]},
        {"upper": [float("nan")]},
        {"nominal_coverage": 0.0},
        {"nominal_coverage": 1.0},
        {"nominal_coverage": True},
    ],
)
def test_interval_invalid_inputs(replacement):
    arguments = {"outcomes": [0.0], "lower": [-1.0], "upper": [1.0], "nominal_coverage": 0.9}
    arguments.update(replacement)
    with pytest.raises(ValidationError):
        score_intervals.Input.model_validate(arguments)


@pytest.mark.parametrize("outcome", [-4.0, 0.0, 0.5, 1.0, 3.0, 8.0])
@pytest.mark.parametrize("sample", [[2.0], [0.0, 2.0], [3.0, -2.0, 0.0, 0.0, 5.0]])
def test_crps_matches_independent_pairwise_formula(outcome, sample, context: OperationContext):
    result = score_empirical_crps.execute(
        score_empirical_crps.Input(outcomes=[outcome], samples=[sample]), context
    )
    # Deliberately quadratic oracle: production integrates CDF gaps instead.
    mean_error = sum(abs(value - outcome) for value in sample) / len(sample)
    mean_distance = sum(abs(a - b) for a in sample for b in sample) / len(sample) ** 2
    assert result.mean_crps == pytest.approx(mean_error - mean_distance / 2)
    assert result.mean_crps >= 0.0


def test_crps_known_values_and_unequal_ensemble_sizes(context: OperationContext):
    result = score_empirical_crps.execute(
        score_empirical_crps.Input(outcomes=[1.0, 5.0], samples=[[0.0, 2.0], [2.0]]), context
    )
    assert result.crps_by_observation == pytest.approx([0.5, 3.0])
    assert result.mean_crps == pytest.approx(1.75)
    assert result.minimum_sample_count == 1
    assert result.maximum_sample_count == 2


def test_crps_invariant_under_permutation_and_sample_replication(context: OperationContext):
    samples = [[-2.0, 0.0, 3.0], [3.0, -2.0, 0.0], [-2.0, 0.0, 3.0] * 4]
    result = score_empirical_crps.execute(
        score_empirical_crps.Input(outcomes=[1.0] * 3, samples=samples), context
    )
    assert result.crps_by_observation == pytest.approx([result.mean_crps] * 3)


def test_crps_translation_and_positive_scaling(context: OperationContext):
    result = score_empirical_crps.execute(
        score_empirical_crps.Input(
            outcomes=[1.0, 101.0, 3.0],
            samples=[[-2.0, 0.0, 3.0], [98.0, 100.0, 103.0], [-6.0, 0.0, 9.0]],
        ),
        context,
    )
    first, translated, scaled = result.crps_by_observation
    assert first == pytest.approx(translated)
    assert scaled == pytest.approx(3 * first)


@pytest.mark.parametrize(
    "arguments",
    [
        {"outcomes": [], "samples": []},
        {"outcomes": [0.0], "samples": [[]]},
        {"outcomes": [0.0, 1.0], "samples": [[0.0]]},
        {"outcomes": [True], "samples": [[0.0]]},
        {"outcomes": [0.0], "samples": [[True]]},
        {"outcomes": [0.0], "samples": [[float("inf")]]},
        {"outcomes": [0.0], "samples": [[1e101]]},
    ],
)
def test_crps_invalid_inputs(arguments):
    with pytest.raises(ValidationError):
        score_empirical_crps.Input.model_validate(arguments)


def test_crps_total_work_bound():
    with pytest.raises(ValidationError, match="250000"):
        score_empirical_crps.Input(outcomes=[0.0] * 251, samples=[[0.0] * 1000] * 251)
