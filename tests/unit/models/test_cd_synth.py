"""Unit tests for quant_fund.models._cd_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._cd_synth import cd_data, gauss_logpdf


def test_data_deterministic() -> None:
    a = cd_data(seed=2, n=100)
    b = cd_data(seed=2, n=100)
    assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))


def test_logpdf_peak_and_monotone() -> None:
    y = np.array([0.0])
    peak = gauss_logpdf(y, 0.0, 1.0)[0]
    tail = gauss_logpdf(np.array([3.0]), 0.0, 1.0)[0]
    assert peak > tail
    assert peak == pytest.approx(-0.5 * np.log(2 * np.pi))


def test_logpdf_rejects_nonpositive_sd() -> None:
    with pytest.raises(ValueError, match="sd"):
        gauss_logpdf(np.array([0.0]), 0.0, 0.0)
    with pytest.raises(ValueError, match="sd"):
        gauss_logpdf(np.array([0.0]), 0.0, -1.0)
    with pytest.raises(ValueError, match="sd"):
        gauss_logpdf(np.array([0.0]), 0.0, np.nan)
