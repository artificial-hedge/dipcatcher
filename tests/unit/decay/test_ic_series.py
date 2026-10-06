import numpy as np
import pytest

from quant_fund.decay.ic_series import (
    effective_sample_tstat,
    ic_summary,
    ic_tstat,
    pearson_ic,
    spearman_ic,
)

pytestmark = pytest.mark.synthetic


def test_perfect_monotone_pred_gives_one() -> None:
    pred = np.tile(np.arange(1.0, 6.0), (3, 1))
    actual = pred + 0.5
    ic = pearson_ic(pred, actual)
    np.testing.assert_allclose(ic, np.ones(3))


def test_perfect_reverse_gives_minus_one() -> None:
    pred = np.tile(np.arange(1.0, 6.0), (2, 1))
    actual = -pred
    ic = spearman_ic(pred, actual)
    np.testing.assert_allclose(ic, -np.ones(2))


def test_noisy_pred_recovers_sign() -> None:
    rng = np.random.default_rng(30)
    pred = rng.standard_normal((400, 50))
    actual = pred + 3.0 * rng.standard_normal((400, 50))
    ic = spearman_ic(pred, actual)
    assert np.nanmean(ic) > 0.05


def test_constant_row_is_nan() -> None:
    pred = np.ones((4, 5))
    actual = np.arange(20.0).reshape(4, 5)
    ic = pearson_ic(pred, actual)
    assert np.all(np.isnan(ic))


def test_ic_summary_fields() -> None:
    rng = np.random.default_rng(31)
    ic = 0.05 + 0.1 * rng.standard_normal(300)
    out = ic_summary(ic)
    for key in ("n", "mean", "std", "tstat", "ir", "skew", "excess_kurtosis", "min", "max"):
        assert key in out
    assert out["n"] == 300
    assert abs(out["mean"] - 0.05) < 0.02


def test_ic_tstat_scales_with_sqrt_n() -> None:
    rng = np.random.default_rng(32)
    short = 0.1 + 0.2 * rng.standard_normal(100)
    long = 0.1 + 0.2 * rng.standard_normal(400)
    assert abs(ic_tstat(long)) > abs(ic_tstat(short))


def test_effective_sample_tstat_downweights_positive_autocorr() -> None:
    rng = np.random.default_rng(33)
    x = np.empty(800)
    x[0] = 0.1
    e = rng.standard_normal(800)
    for i in range(1, 800):
        x[i] = 0.8 * x[i - 1] + 0.1 + 0.2 * e[i]
    out = effective_sample_tstat(x, max_lag=5)
    assert out["rho"] > 0.5
    assert abs(out["adjusted_tstat"]) < abs(out["tstat"])


def test_shape_validation() -> None:
    with pytest.raises(ValueError):
        pearson_ic(np.zeros((3, 4)), np.zeros((3, 5)))
    with pytest.raises(ValueError):
        ic_summary(np.array([0.1, 0.2]))
