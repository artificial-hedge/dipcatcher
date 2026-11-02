"""Unit tests for the trailing-window operations in ``fx1.operations``.

These three modules (``rolling_mad``, ``rolling_rank``,
``rolling_autocorrelation``) share one contract and were previously untested:

* a row is only reported once its **complete** trailing window exists; earlier
  rows are ``None`` and carry an explicit status, never a fabricated number;
* the window is *current-inclusive* — the current row is part of its own
  window;
* the caller supplies ordered, PIT-selected rows. The transform cannot establish
  point-in-time provenance itself, so the tests fix that as a precondition
  rather than pretending the module establishes it.

The edge cases pinned here are the ones a financial consumer would silently
get wrong: a zero MAD, an even-window median that sits between two
representable floats, and a score that is not representable at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations.base import OperationContext
from fx1.operations.rolling_autocorrelation import Input as AcInput
from fx1.operations.rolling_autocorrelation import execute as ac_execute
from fx1.operations.rolling_mad import Input as MadInput
from fx1.operations.rolling_mad import execute as mad_execute
from fx1.operations.rolling_rank import Input as RankInput
from fx1.operations.rolling_rank import execute as rank_execute

CONTEXT = OperationContext(workspace_root=Path(__file__).resolve())


def _mad(values: list[float], window: int):
    return mad_execute(MadInput(values=values, window=window), CONTEXT)


def _rank(values: list[float], window: int):
    return rank_execute(RankInput(values=values, window=window), CONTEXT)


def _ac(values: list[float], window: int, lag: int = 1, minimum_pairs: int = 2):
    return ac_execute(
        AcInput(
            values=values,
            window=window,
            lag=lag,
            minimum_pairs=minimum_pairs,
        ),
        CONTEXT,
    )


# --------------------------------------------------------------------------- #
# rolling_mad
# --------------------------------------------------------------------------- #


def test_mad_warmup_rows_are_null_and_flagged() -> None:
    """Rows without a complete window report nothing rather than a partial stat."""
    out = _mad([1.0, 2.0, 3.0, 4.0], window=3)
    assert out.medians == [None, None, 2.0, 3.0]
    assert out.median_absolute_deviations == [None, None, 1.0, 1.0]
    assert out.robust_zscores == [None, None, 1.0, 1.0]
    assert out.zscore_status == ["warmup", "warmup", "finite", "finite"]
    assert out.window == 3
    # The contract is an *unscaled* raw MAD: no normality consistency factor.
    assert out.normal_consistency_factor_applied is False


def test_mad_odd_window_median_is_the_middle_observation() -> None:
    out = _mad([10.0, 20.0, 30.0], window=3)
    assert out.medians[2] == pytest.approx(20.0)
    assert out.median_absolute_deviations[2] == pytest.approx(10.0)
    # The current row (30) sits one raw-MAD above the median.
    assert out.robust_zscores[2] == pytest.approx(1.0)


def test_mad_even_window_median_averages_the_two_middle_values() -> None:
    """An even window's median lies between the two central observations."""
    out = _mad([1.0, 3.0], window=2)
    assert out.medians[1] == pytest.approx(2.0)
    assert out.median_absolute_deviations[1] == pytest.approx(1.0)
    # current (3.0) minus center (2.0) over the raw MAD (1.0).
    assert out.robust_zscores[1] == pytest.approx(1.0)
    assert out.zscore_status[1] == "finite"


def test_mad_zero_mad_reports_zero_mad_not_a_score() -> None:
    """A constant window has no dispersion; the score must be null, not 0."""
    out = _mad([5.0, 5.0, 5.0], window=3)
    assert out.median_absolute_deviations[2] == pytest.approx(0.0)
    assert out.robust_zscores[2] is None
    assert out.zscore_status[2] == "zero_mad"


def test_mad_unrepresentable_score_reports_numeric_overflow() -> None:
    """A tiny non-zero MAD with a far-away current value overflows the score."""
    # The smallest positive denormal keeps the MAD non-zero while the current
    # value sits 200 orders of magnitude away — the ratio is not representable.
    out = _mad([0.0, 5e-324, 1e100], window=3)
    assert out.median_absolute_deviations[2] == pytest.approx(5e-324)
    assert out.robust_zscores[2] is None
    assert out.zscore_status[2] == "numeric_overflow"


def test_mad_large_but_representable_score_is_still_finite() -> None:
    """A huge but finite ratio is reported as a number, not as an overflow."""
    out = _mad([0.0, 1e-100, 1e100], window=3)
    assert out.zscore_status[2] == "finite"
    assert out.robust_zscores[2] == pytest.approx(1e200)


def test_mad_window_of_one_is_a_zero_deviation_window() -> None:
    out = _mad([7.0, 7.0], window=1)
    assert out.medians == [7.0, 7.0]
    assert out.median_absolute_deviations == [0.0, 0.0]
    assert out.zscore_status == ["zero_mad", "zero_mad"]


def test_mad_rejects_nonfinite_values() -> None:
    """NaN and infinity never enter the transform."""
    with pytest.raises(ValidationError):
        MadInput(values=[1.0, float("nan"), 3.0], window=3)
    with pytest.raises(ValidationError):
        MadInput(values=[1.0, float("inf"), 3.0], window=3)
    with pytest.raises(ValidationError):
        MadInput(values=[1.0, 2.0], window=0)


def test_mad_accepts_integer_literals_by_coercing_them_to_float() -> None:
    """Integers are widened to float; the transform never sees a mixed dtype."""
    model = MadInput(values=[1, 2, 3], window=3)  # type: ignore[list-item]
    assert model.values == [1.0, 2.0, 3.0]
    assert all(isinstance(value, float) for value in model.values)


# --------------------------------------------------------------------------- #
# rolling_rank
# --------------------------------------------------------------------------- #


def test_rank_warmup_rows_are_null() -> None:
    out = _rank([1.0, 2.0, 3.0], window=3)
    assert out.percentiles == [None, None, 1.0]
    assert out.window == 3
    assert out.tie_convention == "average_one_based_rank_divided_by_window"


def test_rank_unique_maximum_is_one_and_unique_minimum_is_one_over_window() -> None:
    high = _rank([1.0, 2.0], window=2)
    assert high.percentiles[1] == pytest.approx(1.0)

    low = _rank([2.0, 1.0], window=2)
    assert low.percentiles[1] == pytest.approx(0.5)


def test_rank_ties_use_the_average_one_based_rank() -> None:
    """Three tied observations all rank in the middle of the window."""
    out = _rank([1.0, 1.0, 1.0], window=3)
    # smaller = 0, ties = 3 -> (0 + (3 + 1) / 2) / 3 = 2/3.
    assert out.percentiles[2] == pytest.approx(2.0 / 3.0)


def test_rank_partial_tie_uses_the_average_of_the_tied_block() -> None:
    """One lower value and two tied maxima sit at the average of 2 and 3."""
    out = _rank([0.0, 5.0, 5.0], window=3)
    # smaller = 1, ties = 2 -> (1 + (2 + 1) / 2) / 3 = 2.5 / 3.
    assert out.percentiles[2] == pytest.approx(2.5 / 3.0)


def test_rank_window_is_current_inclusive() -> None:
    """The current row belongs to its own window — a lagged rank would be 1/3."""
    out = _rank([1.0, 2.0, 3.0], window=3)
    assert out.percentiles[2] == pytest.approx(1.0)
    lagged = _rank([1.0, 2.0, 3.0], window=3)
    assert lagged.percentiles[2] != pytest.approx(1.0 / 3.0)


def test_rank_rejects_out_of_range_window() -> None:
    with pytest.raises(ValidationError):
        RankInput(values=[1.0], window=0)
    with pytest.raises(ValidationError):
        RankInput(values=[1.0], window=1001)


# --------------------------------------------------------------------------- #
# rolling_autocorrelation
# --------------------------------------------------------------------------- #


def test_autocorrelation_warmup_rows_are_null_and_flagged() -> None:
    out = _ac([1.0, 2.0, 3.0, 4.0], window=4, lag=1)
    assert out.correlations[:3] == [None, None, None]
    assert out.status[:3] == ["warmup", "warmup", "warmup"]
    assert out.window == 4
    assert out.lag == 1
    assert out.pairs_per_full_window == 3


def test_autocorrelation_is_one_for_a_perfectly_linear_series() -> None:
    out = _ac([1.0, 2.0, 3.0, 4.0, 5.0], window=4, lag=1)
    assert out.correlations[3] == pytest.approx(1.0)
    assert out.status[3] == "finite"


def test_autocorrelation_is_minus_one_for_an_alternating_series() -> None:
    """The lagged pairs move in opposite directions, so the correlation is -1."""
    out = _ac([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], window=4, lag=1)
    assert out.correlations[3] == pytest.approx(-1.0)
    assert out.status[3] == "finite"


def test_autocorrelation_of_a_monotonic_series_is_positive() -> None:
    """A steadily falling series correlates +1 with its own lagged pairs."""
    out = _ac([5.0, 4.0, 3.0, 2.0, 1.0], window=4, lag=1)
    assert out.correlations[3] == pytest.approx(1.0)


def test_autocorrelation_zero_variance_window_is_null() -> None:
    out = _ac([2.0, 2.0, 2.0, 2.0], window=3, lag=1)
    assert out.correlations[2] is None
    assert out.status[2] == "zero_variance"


def test_autocorrelation_centers_each_pair_vector_separately() -> None:
    """A shifted window keeps full correlation — a shared-mean estimator would not."""
    shifted = [100.0, 101.0, 102.0, 103.0, 104.0]
    out = _ac(shifted, window=4, lag=1)
    assert out.correlations[3] == pytest.approx(1.0)


def test_autocorrelation_rejects_too_few_pairs() -> None:
    """window - lag must still provide ``minimum_pairs`` observations."""
    with pytest.raises(ValidationError):
        AcInput(values=[1.0, 2.0, 3.0], window=3, lag=2, minimum_pairs=2)
    with pytest.raises(ValidationError):
        AcInput(values=[1.0, 2.0, 3.0], window=3, lag=1, minimum_pairs=3)


def test_autocorrelation_rejects_too_small_window() -> None:
    with pytest.raises(ValidationError):
        AcInput(values=[1.0, 2.0], window=2, lag=1, minimum_pairs=2)
