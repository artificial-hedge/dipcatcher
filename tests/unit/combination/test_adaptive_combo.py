import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.combination.adaptive_combo import (
    adaptive_combine,
    adaptive_vs_static,
    fit_adaptive_weights,
)

pytestmark = pytest.mark.synthetic

TAUS = np.linspace(0.1, 0.9, 9)


def _regime_members(seed: int, t_total: int = 400):
    """Member A good in regime 0, member B good in regime 1."""
    rng = np.random.default_rng(seed)
    regime = (np.arange(t_total) // 100) % 2  # alternating blocks
    y = rng.standard_normal(t_total)
    q = np.empty((t_total, 2, len(TAUS)))
    z = norm.ppf(TAUS)
    for t_i in range(t_total):
        good = 0 if regime[t_i] == 0 else 1
        bad = 1 - good
        q[t_i, good, :] = y[t_i] + 0.1 * rng.standard_normal(len(TAUS))
        q[t_i, bad, :] = y[t_i] + 1.0 * rng.standard_normal(len(TAUS))
        q[t_i, :, :] += z[None, :] * 0.5
    features = regime.astype(np.float64)[:, None]
    return features, q, y, regime


def test_fit_returns_beta_shape() -> None:
    features, q, y, _ = _regime_members(100)
    fit = fit_adaptive_weights(features[:200], q[:200], y[:200], TAUS)
    beta = fit["beta"]
    assert beta.shape == (2, 2)  # [intercept, feature] × members


def test_adaptive_combine_tracks_regime() -> None:
    features, q, y, regime = _regime_members(101)
    fit = fit_adaptive_weights(features[:200], q[:200], y[:200], TAUS)
    out = adaptive_combine(features[200:], q[200:], fit["beta"])
    w = out["weights"]
    after = regime[200:] == 1
    assert np.mean(w[after, 1]) > np.mean(w[after, 0])


def test_adaptive_beats_static_out_of_sample() -> None:
    features, q, y, _ = _regime_members(102)
    out = adaptive_vs_static(features, q, y, TAUS, train_frac=0.5)
    assert out["adaptive_pinball"] < out["static_pinball"]


def test_validation() -> None:
    with pytest.raises(ValueError):
        fit_adaptive_weights(np.zeros((10, 1)), np.zeros((10, 2, 3)), np.zeros(9), TAUS)
    with pytest.raises(ValueError):
        adaptive_vs_static(np.zeros((30, 1)), np.zeros((30, 2, 9)), np.zeros(30), TAUS)
