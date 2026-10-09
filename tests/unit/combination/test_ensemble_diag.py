import numpy as np
import pytest

from quant_fund.combination.ensemble_diag import (
    diversity_report,
    encompassing_test,
    error_correlation,
)

pytestmark = pytest.mark.synthetic


def _members(seed: int, n: int = 3000):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    f1 = y + rng.standard_normal(n) * 0.5
    f2 = f1 + rng.standard_normal(n) * 0.2  # shares f1's noise
    f3 = y + rng.standard_normal(n) * 0.6  # independent noise
    return np.column_stack([f1, f2, f3]), y


def test_error_correlation_structure() -> None:
    f, y = _members(30)
    corr = error_correlation(f, y)
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0)
    assert corr[0, 1] > 0.5  # f1, f2 share noise
    assert corr[0, 1] > corr[0, 2]


def test_encompassing_flags_complementary_member() -> None:
    rng = np.random.default_rng(31)
    n = 4000
    y = rng.standard_normal(n)
    f1 = y + rng.standard_normal(n) * 0.6
    f2 = y + rng.standard_normal(n) * 0.6
    out = encompassing_test(np.column_stack([f1, f2]), np.array([0.5, 0.5]), y)
    # f1 − comb = 0.5(f1 − f2), pure noise → not significant
    assert np.all(np.abs(out["t_gamma"]) < 3.0)
    assert out["r2"] > 0.5


def test_diversity_report_flags_equal_weight_win() -> None:
    rng = np.random.default_rng(32)
    n = 4000
    y = rng.standard_normal(n)
    f1 = y + rng.standard_normal(n) * 0.4
    f2 = y + rng.standard_normal(n) * 0.4 + rng.standard_normal(n) * 0.3
    rep = diversity_report(np.column_stack([f1, f2]), y)
    assert rep["mean_error_corr"] < 0.8
    assert rep["equal_beats_best"] == 1.0
    assert rep["equal_weight_mse"] < rep["best_member_mse"] * 1.05


def test_validation() -> None:
    with pytest.raises(ValueError):
        error_correlation(np.zeros((10, 2)), np.zeros(9))
    with pytest.raises(ValueError):
        encompassing_test(np.zeros((10, 2)), np.zeros(3), np.zeros(10))
