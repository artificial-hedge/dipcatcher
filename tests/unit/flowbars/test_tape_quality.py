import numpy as np
import pytest

from quant_fund.flowbars.tape_quality import (
    discretization_grid,
    gap_time_stats,
    outlier_trade_rate,
    tape_quality_report,
    zero_change_runs,
)

pytestmark = pytest.mark.synthetic


def test_zero_change_runs_flat_tape() -> None:
    p = np.array([10.0, 10.0, 10.0, 10.0, 11.0, 11.0])
    out = zero_change_runs(p)
    assert out["max_run"] == 3.0
    assert out["share_zero"] == pytest.approx(4.0 / 5.0)


def test_zero_change_runs_all_moving() -> None:
    p = np.arange(20, dtype=np.float64)
    out = zero_change_runs(p)
    assert out["max_run"] == 0.0
    assert out["share_zero"] == 0.0


def test_discretization_grid_penny() -> None:
    p = np.array([10.00, 10.01, 10.02, 10.01, 10.03])
    assert discretization_grid(p) == pytest.approx(0.01)


def test_discretization_grid_no_moves() -> None:
    assert discretization_grid(np.full(10, 100.0)) == 0.0


def test_outlier_rate_low_for_clean_tape() -> None:
    rng = np.random.default_rng(50)
    p = np.exp(np.concatenate(([0.0], np.cumsum(rng.normal(0, 0.01, 5000)))))
    assert outlier_trade_rate(p, window=200, k=5.0) < 0.05


def test_outlier_rate_high_for_spiky_tape() -> None:
    rng = np.random.default_rng(51)
    r = rng.normal(0, 0.01, 5000)
    r[2000] = 0.5  # one big spike
    p = np.exp(np.concatenate(([0.0], np.cumsum(r))))
    assert outlier_trade_rate(p, window=200, k=5.0) > 0.0


def test_gap_time_stats() -> None:
    g = np.array([1.0, 2.0, 2.0, 3.0, 8.0])
    out = gap_time_stats(g)
    assert out["median"] == 2.0
    assert out["max"] == 8.0


def test_tape_quality_report_bundle() -> None:
    rng = np.random.default_rng(52)
    p = np.exp(np.concatenate(([0.0], np.cumsum(rng.normal(0, 0.01, 3000)))))
    g = np.abs(rng.normal(1.0, 0.5, 2999))
    out = tape_quality_report(p, g)
    for key in ("max_stale_run", "share_zero_changes", "outlier_rate", "cleanliness"):
        assert key in out
    assert "gap_median" in out
    assert 0.0 <= out["cleanliness"] <= 1.0


def test_validation() -> None:
    with pytest.raises(ValueError):
        zero_change_runs(np.array([1.0]))
    with pytest.raises(ValueError):
        gap_time_stats(np.array([1.0, -0.5]))
