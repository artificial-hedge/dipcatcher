"""Order-statistic pairwise absolute sums.

The integer fixtures are exact: a wrong coefficient on ``(2i - n - 1)`` or a
tie handled on the wrong side of ``searchsorted`` fails them. Float fixtures
compare against the expanded double loop, which is the estimator the identity
replaces. Research arithmetic only.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.scoring import crps_empirical
from quant_fund.metrics.twcrps import crps_ensemble
from quant_fund.metrics.twosample import energy_distance
from quant_fund.utils.absdiff import (
    ensemble_crps,
    mean_cross_abs,
    mean_pairwise_abs,
    sum_cross_abs,
    sum_pairwise_abs,
)
from quant_fund.validation.vintage_eval import _crps_ensemble


def _brute_self(values: np.ndarray) -> float:
    total = 0.0
    for left in values:
        for right in values:
            total += abs(float(left) - float(right))
    return total


def _brute_cross(left: np.ndarray, right: np.ndarray) -> float:
    total = 0.0
    for a in left:
        for b in right:
            total += abs(float(a) - float(b))
    return total


def test_pairwise_abs_integer_identity() -> None:
    # Hand sum for [1, 2, 4] is 12. The weight row is (-2, 0, 2).
    values = np.array([4.0, 1.0, 2.0])
    assert sum_pairwise_abs(values) == 12.0
    assert sum_pairwise_abs(values) == _brute_self(values)


def test_cross_abs_integer_identity_with_ties() -> None:
    # Hand sum of |{1,4} - {2,2,5}| is 11. Ties sit on the left edge.
    left = np.array([4.0, 1.0])
    right = np.array([2.0, 5.0, 2.0])
    assert sum_cross_abs(left, right) == 11.0
    assert sum_cross_abs(left, right) == _brute_cross(left, right)


def test_pairwise_abs_matches_expanded_sum_on_floats() -> None:
    rng = np.random.default_rng(11)
    left = rng.normal(size=36)
    right = rng.normal(size=28)
    assert sum_pairwise_abs(left) == pytest.approx(_brute_self(left), rel=1e-12, abs=1e-9)
    assert sum_cross_abs(left, right) == pytest.approx(
        _brute_cross(left, right), rel=1e-12, abs=1e-9
    )
    n = left.size
    assert mean_pairwise_abs(left) == pytest.approx(sum_pairwise_abs(left) / (n * n))
    assert mean_cross_abs(left, right) == pytest.approx(
        sum_cross_abs(left, right) / (left.size * right.size)
    )


def test_permutation_invariance() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(size=25)
    shuffled = rng.permutation(values)
    assert sum_pairwise_abs(values) == sum_pairwise_abs(shuffled)


def test_ensemble_crps_two_point_and_shared_entry_points() -> None:
    sample = np.array([-1.0, 1.0])
    # E|X-y| = 1, half the pairwise mean is 0.5, CRPS = 0.5.
    assert ensemble_crps(0.0, sample) == 0.5
    assert crps_empirical(0.0, sample) == crps_ensemble(sample, 0.0)
    assert crps_empirical(0.0, sample[::-1]) == 0.5
    vintage = _crps_ensemble(np.array([0.0]), sample.reshape(1, -1))
    assert vintage == crps_empirical(0.0, sample)


def test_energy_distance_identical_integers_are_zero() -> None:
    values = np.array([1.0, 2.0, 3.0, 3.0, 8.0])
    assert energy_distance(values, values.copy()) == 0.0


def test_energy_distance_matches_broadcast() -> None:
    rng = np.random.default_rng(5)
    left = rng.normal(size=24)
    right = rng.normal(size=21) + 0.4
    broadcast = (
        2.0 * float(np.mean(np.abs(left[:, None] - right[None, :])))
        - float(np.mean(np.abs(left[:, None] - left[None, :])))
        - float(np.mean(np.abs(right[:, None] - right[None, :])))
    )
    assert energy_distance(left, right) == pytest.approx(max(broadcast, 0.0), rel=1e-12, abs=1e-12)


def test_absdiff_fail_closed() -> None:
    with pytest.raises(ValueError, match="nonempty finite"):
        sum_pairwise_abs(np.array([]))
    with pytest.raises(ValueError, match="nonempty finite"):
        sum_cross_abs(np.array([1.0]), np.array([np.nan]))
    with pytest.raises(ValueError, match="finite"):
        ensemble_crps(float("nan"), np.array([1.0, 2.0]))
