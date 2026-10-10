import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.combination.quantile_regression_combo import (
    combined_quantiles,
    pinball_optimal_all_levels,
    pinball_optimal_weights,
)

pytestmark = pytest.mark.synthetic

TAUS = np.linspace(0.1, 0.9, 9)


def _members(seed: int, t_total: int = 1500):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(t_total)
    q = np.empty((t_total, 2, len(TAUS)))
    z = norm.ppf(TAUS)
    q[:, 0, :] = y[:, None] + 0.2 * rng.standard_normal((t_total, len(TAUS)))
    q[:, 1, :] = y[:, None] + 1.2 * rng.standard_normal((t_total, len(TAUS)))
    q += z[None, None, :] * 0.5
    return q, y


def test_weights_concentrate_on_better_member() -> None:
    q, y = _members(110)
    w = pinball_optimal_weights(q[:, :, 4], y, float(TAUS[4]))
    assert w[0] > 0.8
    assert abs(float(w.sum()) - 1.0) < 1e-9


def test_weights_on_simplex_when_members_equal() -> None:
    rng = np.random.default_rng(111)
    y = rng.standard_normal(1000)
    q = np.stack([y + rng.standard_normal(1000) * 0.5, y + rng.standard_normal(1000) * 0.5], axis=1)
    w = pinball_optimal_weights(q, y, 0.5)
    assert np.all(w >= -1e-12)
    assert abs(float(w.sum()) - 1.0) < 1e-9


def test_all_levels_shape_and_consistency() -> None:
    q, y = _members(112)
    w = pinball_optimal_all_levels(q, y, TAUS, iters=200)
    assert w.shape == (2, len(TAUS))
    assert np.all(w[0, :] > 0.6)


def test_combined_quantiles_monotone() -> None:
    q, y = _members(113)
    w = pinball_optimal_all_levels(q, y, TAUS, iters=200)
    combined = combined_quantiles(q, w)
    assert combined.shape == (len(y), len(TAUS))
    assert np.all(np.diff(combined, axis=1) >= 0.0)


def test_combined_beats_plain_mean_member() -> None:
    q, y = _members(114)
    w = pinball_optimal_weights(q[:, :, 4], y, float(TAUS[4]))
    combined = q[:, :, 4] @ w
    mean_member = q[:, :, 4].mean(axis=1)

    def pinball(a: np.ndarray, b: np.ndarray, tau: float) -> float:
        d = a - b
        return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))

    tau = float(TAUS[4])
    assert pinball(y, combined, tau) < pinball(y, mean_member, tau)


def test_validation() -> None:
    with pytest.raises(ValueError):
        pinball_optimal_weights(np.zeros((10, 3)), np.zeros(9), 0.5)
    with pytest.raises(ValueError):
        pinball_optimal_weights(np.zeros((10, 3)), np.zeros(10), 1.5)
