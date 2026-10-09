import numpy as np
import pytest

from quant_fund.decay.signal_stability import (
    cusum_break,
    rolling_ic_stats,
    stability_score,
)

pytestmark = pytest.mark.synthetic


def test_rolling_stats_nan_warmup() -> None:
    rng = np.random.default_rng(20)
    ic = 0.05 + 0.1 * rng.standard_normal(200)
    stats = rolling_ic_stats(ic, window=40, min_periods=20)
    assert np.all(np.isnan(stats["mean"][:19]))
    assert np.all(np.isfinite(stats["mean"][19:]))
    assert np.all(np.isfinite(stats["ir"][19:]))


def test_cusum_alarms_after_shift() -> None:
    rng = np.random.default_rng(21)
    x = np.concatenate(
        [0.05 + 0.1 * rng.standard_normal(150), 0.25 + 0.1 * rng.standard_normal(150)]
    )
    out = cusum_break(x, pre_period=100, threshold=5.0)
    assert out["alarm_time"] > 0
    assert out["max_cusum"] > 5.0


def test_cusum_quiet_without_shift() -> None:
    rng = np.random.default_rng(22)
    x = 0.05 + 0.1 * rng.standard_normal(300)
    out = cusum_break(x, pre_period=100, threshold=8.0)
    assert out["alarm_time"] < 0


def test_stability_score_reflects_ir() -> None:
    rng = np.random.default_rng(23)
    strong = 0.12 + 0.08 * rng.standard_normal(300)
    weak = 0.005 + 0.1 * rng.standard_normal(300)
    assert stability_score(strong, window=50, min_periods=30, ref_ir=0.5) > 0.8
    assert stability_score(weak, window=50, min_periods=30, ref_ir=0.5) < 0.3


def test_validation() -> None:
    with pytest.raises(ValueError):
        rolling_ic_stats(np.zeros(10), window=5, min_periods=6)
    with pytest.raises(ValueError):
        cusum_break(np.full(50, 0.1), pre_period=40)
