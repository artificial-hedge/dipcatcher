"""Adversarial probes for gas_score."""

import numpy as np
import pytest

from quant_fund.models import gas_score as gs


def test_fisher_recovers_normal_limit():
    """nu -> inf must give the Gaussian Fisher for sigma^2: 1/(2 theta^2)."""
    theta = 1.7
    got = gs.gas_t_fisher(theta, 1e6)
    expected = 1.0 / (2.0 * theta * theta)
    assert got == pytest.approx(expected, rel=1e-3)


def test_poisson_rejects_noninteger_counts():
    y = np.arange(20, dtype=float) + 0.5
    with pytest.raises(ValueError, match="integer"):
        gs.gas_poisson(y)


def test_synth_gas_t_rejects_nu_le_2():
    with pytest.raises(ValueError, match="nu>2"):
        gs.synth_gas_t(nu=2.0)


def test_mle_rejects_nonfinite_input():
    y = np.ones(50)
    y[17] = np.nan
    with pytest.raises(ValueError, match="finite"):
        gs.gas_t_mle(y)


def test_filter_rejects_fisher_scaling_without_fn():
    y = np.random.default_rng(0).normal(size=20)
    with pytest.raises(ValueError, match="fisher_fn"):
        gs.gas_filter(y, lambda a, b: 0.0, 0.0, 0.1, 0.9, scaling="inv_fisher")
