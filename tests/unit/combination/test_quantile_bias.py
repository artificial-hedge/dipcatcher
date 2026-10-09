import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.combination.quantile_bias import (
    bias_correct_quantiles,
    exceedance_rates,
    pinball_decomposition,
)

pytestmark = pytest.mark.synthetic

TAUS = np.linspace(0.1, 0.9, 9)


def _quantiles(seed: int, bias: float = 0.0, n: int = 4000):
    """Calibrated member: q_τ = μ̂ + σ̂ z_τ with μ̂ = y + noise, σ̂ estimated."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    z = norm.ppf(TAUS)
    mu_hat = y + 0.3 * rng.standard_normal(n)
    sigma_hat = 0.3  # residual sd of y − μ̂
    q = mu_hat[:, None] + sigma_hat * z[None, :] + bias
    return y, q


def test_exceedance_rates_calibrated_member() -> None:
    y, q = _quantiles(40)
    rates = exceedance_rates(y, q, TAUS)
    np.testing.assert_allclose(rates, TAUS, atol=0.03)


def test_exceedance_detects_bias() -> None:
    y, q = _quantiles(41, bias=0.15)
    rates = exceedance_rates(y, q, TAUS)
    # upward shift raises exceedance above nominal at every level
    assert np.all(rates > TAUS + 0.05)


def test_pinball_decomposition_components() -> None:
    y, q = _quantiles(42)
    out = pinball_decomposition(y, q, TAUS)
    assert out["total"] > 0
    assert out["reliability"] >= 0
    assert out["sharpness"] <= out["total"] + 1e-12
    assert abs(out["exceedance"] - 0.5) < 0.05


def test_bias_correction_fixes_exceedance() -> None:
    y, q = _quantiles(43, bias=0.5)
    before = exceedance_rates(y, q, TAUS)
    out = bias_correct_quantiles(q, TAUS, y=y)
    after = exceedance_rates(y, out["quantiles"], TAUS)
    assert np.abs(after[4] - 0.5) < np.abs(before[4] - 0.5)
    assert np.all(np.diff(out["quantiles"], axis=1) >= 0.0)


def test_validation() -> None:
    with pytest.raises(ValueError):
        exceedance_rates(np.zeros(5), np.zeros((5, 3)), np.array([0.2, 0.5]))
    with pytest.raises(ValueError):
        bias_correct_quantiles(np.zeros((5, 3)), np.array([0.2, 0.5, 0.8]))
