"""Unit tests for quant_fund.models.cos_method."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.models.cos_method import (
    bench_cos_method,
    bs_charfn,
    cos_call,
    cos_put,
)


def _bs(s0, k, t, r, sigma):
    sd = sigma * np.sqrt(t)
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma * sigma) * t) / sd
    d2 = d1 - sd
    return s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


def test_cos_matches_black() -> None:
    s0, k, t, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    mu = np.log(s0 / k) + (r - 0.5 * sigma * sigma) * t
    var = sigma * sigma * t
    cums = (mu, var, 3 * var * var)
    cf = lambda u: bs_charfn(np.asarray(u), s0, k, t, r, sigma)  # noqa: E731
    v = cos_call(cf, s0, k, t, r, cumulants=cums)
    assert v == pytest.approx(_bs(s0, k, t, r, sigma), rel=1e-6)


def test_put_parity() -> None:
    s0, k, t, r, sigma = 100.0, 95.0, 0.5, 0.03, 0.3
    mu = np.log(s0 / k) + (r - 0.5 * sigma * sigma) * t
    var = sigma * sigma * t
    cums = (mu, var, 3 * var * var)
    cf = lambda u: bs_charfn(np.asarray(u), s0, k, t, r, sigma)  # noqa: E731
    c = cos_call(cf, s0, k, t, r, cumulants=cums)
    p = cos_put(cf, s0, k, t, r, cumulants=cums)
    assert c - p == pytest.approx(s0 - k * np.exp(-r * t), rel=1e-6)


def test_deep_itm_sanity() -> None:
    s0, k, t, r, sigma = 150.0, 50.0, 1.0, 0.02, 0.2
    mu = np.log(s0 / k) + (r - 0.5 * sigma * sigma) * t
    var = sigma * sigma * t
    cums = (mu, var, 3 * var * var)
    cf = lambda u: bs_charfn(np.asarray(u), s0, k, t, r, sigma)  # noqa: E731
    v = cos_call(cf, s0, k, t, r, cumulants=cums)
    assert v > s0 - k * np.exp(-r * t)


def test_input_validation() -> None:
    cf = lambda u: np.ones(len(np.asarray(u)), dtype=complex)  # noqa: E731
    with pytest.raises(ValueError):
        cos_call(cf, 100.0, 0.0, 1.0, 0.0, cumulants=(0.0, 0.1, 0.1))
    with pytest.raises(ValueError):
        cos_call(cf, 100.0, 100.0, 1.0, 0.0, cumulants=(0.0, -0.1, 0.1))


def test_bench_score() -> None:
    out = bench_cos_method()
    assert out["score"] == 1.0
    assert out["synthetic_cos_call_err"] < 0.01
