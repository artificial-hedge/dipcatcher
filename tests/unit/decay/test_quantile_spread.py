import numpy as np
import pytest

from quant_fund.decay.quantile_spread import (
    bucket_returns,
    bucket_trend_ic,
    monotonicity_score,
    spread_curve,
)

pytestmark = pytest.mark.synthetic


def _monotone_panel(seed: int, t_total: int = 300, n: int = 100):
    rng = np.random.default_rng(seed)
    pred = rng.standard_normal((t_total, n))
    fwd = np.empty((t_total, n))
    for t in range(t_total):
        ranks = np.argsort(np.argsort(pred[t]))
        fwd[t] = 0.002 * ranks + rng.standard_normal(n) * 0.5
    return pred, fwd


def test_bucket_returns_shape() -> None:
    pred, fwd = _monotone_panel(50)
    buckets = bucket_returns(pred, fwd, k=5)
    assert buckets.shape == (300, 5)


def test_monotone_signal_gives_positive_spread() -> None:
    pred, fwd = _monotone_panel(51)
    buckets = bucket_returns(pred, fwd, k=5)
    spread = spread_curve(buckets)
    assert float(np.nanmean(spread)) > 0.1


def test_monotonicity_score_high_for_monotone_signal() -> None:
    pred, fwd = _monotone_panel(52)
    buckets = bucket_returns(pred, fwd, k=5)
    assert monotonicity_score(buckets) > 0.6


def test_bucket_trend_ic_positive() -> None:
    pred, fwd = _monotone_panel(53)
    buckets = bucket_returns(pred, fwd, k=5)
    out = bucket_trend_ic(buckets)
    assert out["trend_ic"] > 0.5


def test_reverse_signal_gives_negative_spread() -> None:
    pred, fwd = _monotone_panel(54)
    buckets = bucket_returns(-pred, fwd, k=5)
    assert float(np.nanmean(spread_curve(buckets))) < -0.1


def test_validation() -> None:
    with pytest.raises(ValueError):
        bucket_returns(np.zeros((5, 3)), np.zeros((5, 3)), k=5)
    with pytest.raises(ValueError):
        bucket_trend_ic(np.zeros((5, 2)))
