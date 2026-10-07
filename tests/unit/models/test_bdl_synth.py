"""Unit tests for quant_fund.models._bdl_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._bdl_synth import bdl_data, coverage, nll_gauss


def test_data_deterministic_and_ood_separated() -> None:
    a = bdl_data(seed=3)
    b = bdl_data(seed=3)
    assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))
    x, _y, _xt, _yt, x_ood = a
    assert np.all(np.abs(x_ood) > 2.5)
    assert np.all(np.abs(x) <= 2.0)


def test_nll_gauss_finite_and_ordered() -> None:
    y = np.array([0.0, 1.0])
    mu = np.array([0.0, 1.0])
    good = nll_gauss(y, mu, np.array([0.1, 0.1]))
    bad = nll_gauss(y, mu, np.array([5.0, 5.0]))
    assert np.isfinite(good) and np.isfinite(bad)
    assert good < bad


def test_coverage_rejects_negative_sd() -> None:
    y = np.array([0.0, 0.5])
    mu = np.array([0.0, 0.5])
    with pytest.raises(ValueError, match="non-negative"):
        coverage(y, mu, np.array([-1.0, 1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        coverage(y, mu, np.array([1.0, 1.0]), z=-1.0)


def test_coverage_bounds() -> None:
    y = np.array([0.0, 10.0])
    mu = np.array([0.0, 0.0])
    sd = np.array([1.0, 1.0])
    c = coverage(y, mu, sd)
    assert c == 0.5  # one inside, one outside
