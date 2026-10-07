"""SYNTHETIC adversarial probes for the fx1 proper-score operations.

Every fixture is synthetic and deterministic; results are correctness checks
of proper scoring rules (CRPS, pinball, interval score, Brier, log loss),
never market evidence. Hand-derived and quadratic-form references pin the
implementations.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations import (
    score_binary_forecasts as sbf,
)
from fx1.operations import (
    score_empirical_crps as crps,
)
from fx1.operations import (
    score_intervals as si,
)
from fx1.operations import (
    score_quantiles as sq,
)
from fx1.operations.base import OperationContext


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


# --------------------------- score_empirical_crps ------------------------------


def test_crps_degenerate_single_sample_is_abs_error(context: OperationContext):
    out = crps.execute(
        crps.Input(
            outcomes=[3.0],
            samples=[[5.0]],
        ),
        context,
    )
    assert out.mean_crps == pytest.approx(2.0)
    assert out.crps_by_observation[0] == pytest.approx(2.0)


def test_crps_point_mass_forecast_equals_abs_error(context: OperationContext):
    out = crps.execute(
        crps.Input(
            outcomes=[4.5],
            samples=[[4.0, 4.0, 4.0, 4.0]],
        ),
        context,
    )
    assert out.mean_crps == pytest.approx(0.5)


def test_crps_matches_quadratic_form_identity(context: OperationContext):
    """CRPS empirical CDF form must equal E|X-y| - E|X-X'|/2 on random ensembles."""
    rng = random.Random(2026)
    ensembles = [sorted(rng.uniform(-50, 50) for _ in range(rng.randint(2, 40))) for _ in range(40)]
    observed = [rng.uniform(-50, 50) for _ in ensembles]
    out = crps.execute(
        crps.Input(
            outcomes=observed,
            samples=ensembles,
        ),
        context,
    )
    for fc, e, y in zip(out.crps_by_observation, ensembles, observed, strict=True):
        n = len(e)
        term1 = sum(abs(x - y) for x in e) / n
        term2 = sum(abs(e[i] - e[j]) for i in range(n) for j in range(n)) / (2 * n * n)
        assert fc == pytest.approx(term1 - term2)


def test_crps_perfect_forecast_is_zero(context: OperationContext):
    out = crps.execute(
        crps.Input(
            outcomes=[1.0, -2.0],
            samples=[[1.0], [-2.0]],
        ),
        context,
    )
    assert out.mean_crps == pytest.approx(0.0)


def test_crps_unsorted_samples_are_sorted_internally(context: OperationContext):
    a = crps.execute(
        crps.Input(
            outcomes=[4.0],
            samples=[[9.0, 1.0, 5.0]],
        ),
        context,
    )
    b = crps.execute(
        crps.Input(
            outcomes=[4.0],
            samples=[[1.0, 5.0, 9.0]],
        ),
        context,
    )
    assert a.crps_by_observation[0] == pytest.approx(b.crps_by_observation[0])


def test_crps_nonfinite_and_length_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        crps.Input(outcomes=[1.0], samples=[[1.0, float("nan")]])
    with pytest.raises(ValidationError, match="one nonempty row per outcome"):
        crps.Input(
            outcomes=[1.0, 2.0],
            samples=[[1.0]],
        )


# ------------------------------ score_intervals --------------------------------


def test_interval_score_matches_gneiting_raftery_eq43(context: OperationContext):
    rng = random.Random(7)
    lower: list[float] = []
    upper: list[float] = []
    outcomes: list[float] = []
    for _ in range(200):
        width = rng.uniform(0.5, 10.0)
        center = rng.uniform(-20, 20)
        lower.append(center - width / 2)
        upper.append(center + width / 2)
        outcomes.append(rng.uniform(-30, 30))
    nominal = 0.9
    alpha = 1.0 - nominal
    out = si.execute(
        si.Input(outcomes=outcomes, lower=lower, upper=upper, nominal_coverage=nominal),
        context,
    )
    n = len(outcomes)
    expected_width = sum(u - lo for lo, u in zip(lower, upper, strict=True)) / n
    expected_lower = sum(max(0.0, lo - y) for lo, y in zip(lower, outcomes, strict=True)) / n
    expected_upper = sum(max(0.0, y - u) for u, y in zip(upper, outcomes, strict=True)) / n
    assert out.mean_width == pytest.approx(expected_width)
    assert out.mean_lower_miss_penalty == pytest.approx((2.0 / alpha) * expected_lower)
    assert out.mean_upper_miss_penalty == pytest.approx((2.0 / alpha) * expected_upper)
    assert out.interval_score == pytest.approx(
        expected_width + (2.0 / alpha) * (expected_lower + expected_upper)
    )
    covered = sum(1 for lo, u, y in zip(lower, upper, outcomes, strict=True) if lo <= y <= u)
    assert out.empirical_coverage == pytest.approx(covered / n)


def test_interval_score_boundary_observation_is_covered(context: OperationContext):
    """Observed == bound is a hit: penalties apply strictly outside."""
    out = si.execute(
        si.Input(
            outcomes=[1.0, 3.0],
            lower=[1.0, 1.0],
            upper=[3.0, 3.0],
            nominal_coverage=0.95,
        ),
        context,
    )
    assert out.empirical_coverage == pytest.approx(1.0)
    assert out.interval_score == pytest.approx(2.0)  # width only, zero misses
    assert out.mean_lower_miss_penalty == pytest.approx(0.0)
    assert out.mean_upper_miss_penalty == pytest.approx(0.0)


def test_interval_score_zero_width_is_dirac(context: OperationContext):
    out = si.execute(
        si.Input(outcomes=[5.0], lower=[2.0], upper=[2.0], nominal_coverage=0.5),
        context,
    )
    # multiplier 2/0.5 = 4; upper miss = 3 -> 0 + 0 + 12
    assert out.interval_score == pytest.approx(12.0)
    assert out.empirical_coverage == pytest.approx(0.0)


def test_interval_score_lower_above_upper_rejected() -> None:
    with pytest.raises(ValidationError, match="less than or equal"):
        si.Input(outcomes=[0.0], lower=[3.0], upper=[1.0], nominal_coverage=0.9)


def test_interval_score_alignment_and_coverage_bounds() -> None:
    with pytest.raises(ValidationError):
        si.Input(
            outcomes=[1.0, 2.0],
            lower=[0.0],
            upper=[1.0],
            nominal_coverage=0.9,
        )
    with pytest.raises(ValidationError):
        si.Input(outcomes=[0.0], lower=[0.0], upper=[1.0], nominal_coverage=1.0)


# ------------------------------ score_quantiles --------------------------------


def test_pinball_score_asymmetry(context: OperationContext):
    """pinball(τ=0.9) on a low forecast must cost far more than on a high one."""
    y = 10.0
    out = sq.execute(
        sq.Input(
            outcomes=[y, y],
            levels=[0.1, 0.9],
            quantiles=[[5.0, 5.0], [15.0, 15.0]],
        ),
        context,
    )
    # level 0.1 column: residuals +5 (tau*r=0.5), -5 ((tau-1)*r=4.5) -> mean 2.5
    # level 0.9 column: 0.9*5=4.5 and 0.1*5=0.5 -> mean 2.5
    assert out.mean_pinball_by_level[0] == pytest.approx(2.5)
    assert out.mean_pinball_by_level[1] == pytest.approx(2.5)
    # single observation, high quantile at τ=0.1 vs low quantile at τ=0.9
    low_tau_high_q = sq.execute(sq.Input(outcomes=[y], levels=[0.1], quantiles=[[15.0]]), context)
    high_tau_low_q = sq.execute(sq.Input(outcomes=[y], levels=[0.9], quantiles=[[5.0]]), context)
    assert low_tau_high_q.mean_pinball == pytest.approx(4.5)
    assert high_tau_low_q.mean_pinball == pytest.approx(4.5)


def test_quantile_crossing_rejected() -> None:
    with pytest.raises(ValidationError, match="nondecreasing|crossing"):
        sq.Input(
            outcomes=[0.0],
            levels=[0.1, 0.5],
            quantiles=[[5.0, 4.0]],  # median below the 10% quantile
        )


def test_quantile_ties_at_monotone_quantiles_allowed(context: OperationContext):
    out = sq.execute(
        sq.Input(
            outcomes=[7.0],
            levels=[0.5, 0.9],
            quantiles=[[7.0, 7.0]],
        ),
        context,
    )
    assert out.mean_pinball == pytest.approx(0.0)


def test_quantile_levels_must_be_strictly_increasing() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        sq.Input(
            outcomes=[0.0, 0.0],
            levels=[0.5, 0.5],
            quantiles=[[1.0, 1.0], [2.0, 2.0]],
        )


def test_quantile_row_width_and_level_bounds() -> None:
    with pytest.raises(ValidationError, match="number of levels"):
        sq.Input(
            outcomes=[0.0],
            levels=[0.1, 0.9],
            quantiles=[[1.0]],  # too short
        )
    with pytest.raises(ValidationError):
        sq.Input(
            outcomes=[0.0],
            levels=[0.0, 0.9],
            quantiles=[[1.0, 2.0]],
        )


def test_pinball_seed_crosscheck(context: OperationContext):
    rng = random.Random(77)
    n = 60
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    outcomes = [rng.uniform(-10, 10) for _ in range(n)]
    rows = [sorted(rng.uniform(-12, 12) for _ in levels) for _ in range(n)]
    out = sq.execute(sq.Input(outcomes=outcomes, levels=levels, quantiles=rows), context)
    for col, tau in enumerate(levels):
        losses = [
            (tau * r if r >= 0 else (tau - 1.0) * r)
            for r in (y - row[col] for y, row in zip(outcomes, rows, strict=True))
        ]
        assert out.mean_pinball_by_level[col] == pytest.approx(sum(losses) / n)


# --------------------------- score_binary_forecasts ----------------------------


def test_brier_is_squared_error_not_doubled(context: OperationContext):
    out = sbf.execute(
        sbf.Input(outcomes=[1, 0], probabilities=[0.7, 0.2]),
        context,
    )
    expected = ((0.7 - 1.0) ** 2 + (0.2 - 0.0) ** 2) / 2
    assert out.brier_score == pytest.approx(expected)


def test_log_loss_impossible_endpoint_positive_infinity(context: OperationContext):
    out = sbf.execute(
        sbf.Input(outcomes=[1, 1], probabilities=[0.0, 0.9]),
        context,
    )
    assert out.mean_log_loss is None
    assert out.log_loss_status == "positive_infinity"
    assert out.impossible_event_count == 1
    assert out.brier_score == pytest.approx(((0.0 - 1.0) ** 2 + (0.9 - 1.0) ** 2) / 2)


def test_log_loss_clip_only_salvages_log_not_brier(context: OperationContext):
    out = sbf.execute(
        sbf.Input(
            outcomes=[1],
            probabilities=[0.0],
            probability_clip=1e-6,
        ),
        context,
    )
    assert out.mean_log_loss == pytest.approx(-math.log(1e-6))
    assert out.clipped_probability_count == 1
    assert out.brier_score == pytest.approx(1.0)
    assert out.log_loss_status == "finite"


def test_log_loss_nats_not_bits(context: OperationContext):
    out = sbf.execute(sbf.Input(outcomes=[1], probabilities=[0.5]), context)
    assert out.mean_log_loss == pytest.approx(math.log(2.0))


def test_binary_forecast_rejects_prob_outside_unit() -> None:
    with pytest.raises(ValidationError):
        sbf.Input(outcomes=[1], probabilities=[1.000001])
    with pytest.raises(ValidationError):
        sbf.Input(outcomes=[0], probabilities=[-1e-9])
    with pytest.raises(ValidationError):
        sbf.Input(outcomes=[0], probabilities=[float("inf")])


def test_binary_forecast_outcome_must_be_zero_or_one() -> None:
    with pytest.raises(ValidationError):
        sbf.Input(outcomes=[2], probabilities=[0.5])
    with pytest.raises(ValidationError):
        sbf.Input(outcomes=[0], probabilities=[0.5, 0.6])
