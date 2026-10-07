"""Adversarial probes for Bai-Perron break detection."""

import numpy as np
import pytest

from quant_fund.models import bai_perron as bp


def _step_data(t: int, b: int, hi_val: float = 10.0):
    x = np.ones((t, 1))
    y = np.zeros(t)
    y[b:] = hi_val
    return y, x


def test_dp_reaches_min_length_final_segment():
    """Break at t-h (final segment exactly min length h) must be reachable."""
    t, trim = 30, 0.15
    h = max(2, int(np.floor(t * trim)))  # = 4
    y, x = _step_data(t, t - h)
    out = bp.breakpoints_dp(y, x, 1, trim=trim)
    assert out["breaks"][0] == t - h
    assert out["ssr_total"][0] < 1e-12


def test_refit_rejects_zero_break():
    y, x = _step_data(30, 15)
    with pytest.raises(ValueError, match="interior"):
        bp.refit_segments(y, x, np.array([0]))


def test_sequential_rejects_zero_mmax():
    y, x = _step_data(30, 15)
    with pytest.raises(ValueError, match="m_max"):
        bp.sequential_breaks(y, x, m_max=0)
