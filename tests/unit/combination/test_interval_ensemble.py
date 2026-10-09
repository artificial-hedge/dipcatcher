import numpy as np
import pytest

from quant_fund.combination.interval_ensemble import (
    band_coverage,
    bonferroni_union,
    quantile_band_average,
    winkler_score,
)

pytestmark = pytest.mark.synthetic


def test_bonferroni_union_bounds() -> None:
    lo = np.array([[0.0, 1.0], [2.0, 0.5]])
    hi = np.array([[1.0, 3.0], [4.0, 2.0]])
    out = bonferroni_union(lo, hi)
    np.testing.assert_allclose(out["lower"], [0.0, 0.5])
    np.testing.assert_allclose(out["upper"], [4.0, 3.0])


def test_quantile_band_average_weighted() -> None:
    lo = np.array([[0.0, 0.0], [2.0, 2.0]])
    hi = np.array([[1.0, 1.0], [3.0, 3.0]])
    out = quantile_band_average(lo, hi, weights=np.array([0.75, 0.25]))
    np.testing.assert_allclose(out["lower"], [0.5, 0.5])
    np.testing.assert_allclose(out["upper"], [1.5, 1.5])


def test_band_coverage_counts() -> None:
    y = np.array([0.0, 5.0, 2.0])
    lo = np.array([-1.0, 3.0, 4.0])
    hi = np.array([1.0, 4.0, 5.0])
    # only y=0 sits inside its band
    assert band_coverage(y, lo, hi) == pytest.approx(1.0 / 3.0)


def test_winkler_score_rewards_coverage_and_penalises_misses() -> None:
    y = np.array([0.0])
    lo_hit = np.array([-1.0])
    hi_hit = np.array([1.0])
    lo_miss = np.array([2.0])
    hi_miss = np.array([3.0])
    hit = winkler_score(y, lo_hit, hi_hit, alpha=0.1)
    miss = winkler_score(y, lo_miss, hi_miss, alpha=0.1)
    assert hit[0] == pytest.approx(2.0)  # width only
    assert miss[0] > 2.0  # width + distance penalty


def test_union_coverage_at_least_member_min() -> None:
    rng = np.random.default_rng(90)
    n = 2000
    y = rng.standard_normal(n)
    m1_lo, m1_hi = y - 2.0, y + 2.0  # ~95.45% each
    m2_lo, m2_hi = y - 1.5, y + 1.5
    out = bonferroni_union(np.stack([m1_lo, m2_lo]), np.stack([m1_hi, m2_hi]))
    cov = band_coverage(y, out["lower"], out["upper"])
    assert cov >= 0.95


def test_validation() -> None:
    with pytest.raises(ValueError):
        bonferroni_union(np.zeros((2, 3)), np.zeros((2, 4)))
    with pytest.raises(ValueError):
        winkler_score(np.zeros(2), np.array([1.0, 0.0]), np.array([2.0, 3.0]), alpha=1.5)
