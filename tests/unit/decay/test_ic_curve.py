import numpy as np
import pytest

from quant_fund.decay.ic_curve import (
    curve_auc,
    forward_returns_matrix,
    ic_curve,
    lag_ic,
    peak_lag,
)

pytestmark = pytest.mark.synthetic


def test_forward_returns_compound() -> None:
    r = np.array([0.01, 0.02, 0.03, 0.04])
    fwd = forward_returns_matrix(r, max_lag=2)
    assert fwd.shape == (2, 2)
    # window ret[t+1..t+k]: fwd[0,1] = r[1], fwd[0,2] = (1.02)(1.03) − 1
    np.testing.assert_allclose(fwd[0, 0], 0.02)
    np.testing.assert_allclose(fwd[0, 1], (1.02 * 1.03) - 1.0)
    np.testing.assert_allclose(fwd[1, 1], (1.03 * 1.04) - 1.0)


def test_signal_peaks_at_its_true_horizon() -> None:
    rng = np.random.default_rng(50)
    t_total, n, max_lag, true_lag = 600, 40, 5, 3
    noise = rng.standard_normal((t_total, n))
    # return at t depends on the signal from t - true_lag
    signal = rng.standard_normal((t_total, n))
    ret = np.zeros((t_total, n))
    for t in range(true_lag, t_total):
        # small magnitudes keep log1p compounding valid; the common positive
        # scale per row does not change cross-sectional ranks (Spearman IC)
        ret[t] = 0.01 * (0.5 * signal[t - true_lag] + noise[t])
    t0 = true_lag  # first index whose forward window is fully valid
    usable = t_total - max_lag - t0 - 1
    fwd = np.empty((usable, max_lag, n))
    log1p = np.log1p(ret)
    cums = np.vstack([np.zeros((1, n)), np.cumsum(log1p, axis=0)])
    for i, t in enumerate(range(t0, t0 + usable)):
        # same convention as forward_returns_matrix: window ret[t+1..t+k]
        fwd[i, :] = np.expm1(cums[t + 2 : t + 2 + max_lag] - cums[t + 1])
    curve = ic_curve(signal[t0 : t0 + usable], fwd)
    assert curve.shape == (max_lag,)
    assert peak_lag(curve) == true_lag


def test_lag_ic_single() -> None:
    rng = np.random.default_rng(51)
    pred = rng.standard_normal(30)
    actual = pred + 0.1 * rng.standard_normal(30)
    assert lag_ic(pred, actual) > 0.3


def test_curve_auc_bounds() -> None:
    flat = np.full(5, 0.1)
    assert curve_auc(flat) == pytest.approx(0.1)
    decaying = 0.2 * 0.5 ** np.arange(1, 6)
    assert 0 < curve_auc(decaying) < 0.2


def test_peak_lag_first_on_monotone_decay() -> None:
    assert peak_lag(np.array([0.3, 0.2, 0.1, 0.05])) == 1


def test_forward_matrix_validates() -> None:
    with pytest.raises(ValueError):
        forward_returns_matrix(np.zeros((3, 3)), 2)
    with pytest.raises(ValueError):
        forward_returns_matrix(np.zeros(5), 0)
