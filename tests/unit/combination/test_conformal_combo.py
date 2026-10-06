import numpy as np
import pytest

from quant_fund.combination.conformal_combo import (
    empirical_coverage,
    split_conformal_intervals,
)

pytestmark = pytest.mark.synthetic


def test_coverage_guarantee_holds_iid() -> None:
    rng = np.random.default_rng(130)
    n_cal, n_test = 2000, 3000
    pred_cal = rng.standard_normal(n_cal)
    y_cal = pred_cal + rng.standard_normal(n_cal)
    pred_test = rng.standard_normal(n_test)
    y_test = pred_test + rng.standard_normal(n_test)
    out = split_conformal_intervals(np.abs(y_cal - pred_cal), pred_test, 0.1)
    cov = empirical_coverage(y_test, out["lower"], out["upper"])
    assert cov >= 0.9 - 0.02


def test_width_reflects_member_quality() -> None:
    rng = np.random.default_rng(131)
    n = 1500
    pred_cal = rng.standard_normal(n)
    y_cal = pred_cal + rng.standard_normal(n)
    pred_test = rng.standard_normal(10)
    good = split_conformal_intervals(np.abs(y_cal - pred_cal), pred_test, 0.1)
    noisy = split_conformal_intervals(5.0 * np.abs(y_cal - pred_cal), pred_test, 0.1)
    assert float(good["q"]) < float(noisy["q"])


def test_finite_sample_correction_uses_ceil() -> None:
    scores = np.arange(1.0, 101.0)  # n=100
    pred = np.zeros(3)
    out = split_conformal_intervals(scores, pred, 0.05)
    # k = ceil(101 × 0.95) = 96 → q = 96th order statistic = 96.0
    assert float(out["q"]) == pytest.approx(96.0)


def test_deterministic_and_symmetric() -> None:
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([10.0, 20.0])
    a = split_conformal_intervals(scores, pred, 0.25)
    b = split_conformal_intervals(scores, pred, 0.25)
    np.testing.assert_array_equal(a["lower"], b["lower"])
    np.testing.assert_allclose(a["upper"] - pred, pred - a["lower"])


def test_empirical_coverage_counts_correctly() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0])
    lo = np.array([-1.0, 0.5, 1.5, 4.0])
    hi = np.array([1.0, 1.5, 2.5, 5.0])
    assert empirical_coverage(y, lo, hi) == pytest.approx(0.75)


def test_validation() -> None:
    with pytest.raises(ValueError):
        split_conformal_intervals(np.array([1.0]), np.zeros(2), 0.1)
    with pytest.raises(ValueError):
        split_conformal_intervals(np.ones(5), np.zeros(2), 1.5)
