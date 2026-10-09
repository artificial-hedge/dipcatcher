import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.combination.conformal_quantiles import (
    cqr_correct,
    cqr_coverage,
    cqr_intervals,
    cqr_scores,
)

pytestmark = pytest.mark.synthetic


def _split(seed: int, n: int = 4000):
    """Member emitting a genuine 80% band (z_0.1..z_0.9 covers 80%)."""
    rng = np.random.default_rng(seed)
    n_cal = n // 2
    y = rng.standard_normal(n)
    z_lo, z_hi = norm.ppf(0.1), norm.ppf(0.9)
    mu_hat = y + 0.3 * rng.standard_normal(n)
    sigma_hat = 0.3  # residual sd of y − μ̂: a genuine 80% band
    lo = mu_hat + sigma_hat * z_lo
    hi = mu_hat + sigma_hat * z_hi
    return y[:n_cal], lo[:n_cal], hi[:n_cal], lo[n_cal:], hi[n_cal:], y[n_cal:]


def test_scores_nonnegative_sided() -> None:
    y = np.array([0.0, 2.0])
    lo = np.array([1.0, 0.0])
    hi = np.array([1.5, 1.0])
    sc = cqr_scores(y, lo, hi)
    np.testing.assert_allclose(sc["lower"], [1.0, 0.0])
    np.testing.assert_allclose(sc["upper"], [0.0, 1.0])


def test_cqr_restores_coverage() -> None:
    y_cal, lo_cal, hi_cal, lo_new, hi_new, y_new = _split(50)
    # two-sided 90% band → per-side 5% correction (Bonferroni split)
    corr = cqr_correct(y_cal, lo_cal, hi_cal, alpha=0.05)
    band = cqr_intervals(lo_new, hi_new, corr)
    cov = cqr_coverage(y_new, band["lower"], band["upper"])
    assert cov >= 0.9 - 0.02
    # correction widened the band
    assert np.mean(band["upper"] - band["lower"]) > np.mean(hi_new - lo_new)


def test_cqr_accepts_valid_band() -> None:
    rng = np.random.default_rng(51)
    y = rng.standard_normal(2000)
    lo = y - 2.0
    hi = y + 2.0
    corr = cqr_correct(y[:1000], lo[:1000], hi[:1000], alpha=0.05)
    assert corr["lower_shift"] <= 0.1
    assert corr["upper_shift"] <= 0.1
    band = cqr_intervals(lo[1000:], hi[1000:], corr)
    assert cqr_coverage(y[1000:], band["lower"], band["upper"]) >= 0.95


def test_validation() -> None:
    with pytest.raises(ValueError):
        cqr_correct(np.zeros(5), np.zeros(5), np.zeros(5), alpha=1.5)
    with pytest.raises(ValueError):
        cqr_coverage(np.zeros(3), np.zeros(4), np.zeros(4))
