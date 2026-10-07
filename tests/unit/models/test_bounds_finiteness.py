"""Probes: partial-identification bounds must fail closed on non-finite
observed outcomes instead of propagating NaN into lb/ub/CI."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.bounds import lee_bounds, manski_bounds, manski_mar_bounds


def _mk(n: int = 200, seed: int = 5):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    m = np.zeros(n)
    m[:20] = 1.0
    return y, m


def test_manski_rejects_nan_observed() -> None:
    y, m = _mk()
    y[50] = np.nan  # an *observed* row
    with pytest.raises(ValueError, match="non-finite"):
        manski_bounds(y, m)


def test_manski_rejects_inf_observed() -> None:
    y, m = _mk()
    y[51] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        manski_bounds(y, m)


def test_mar_rejects_nan_observed() -> None:
    y, m = _mk()
    y[52] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        manski_mar_bounds(y, m)


def test_lee_rejects_nan_in_selected_arm() -> None:
    rng = np.random.default_rng(9)
    n = 400
    t = (rng.random(n) < 0.5).astype(float)
    s = (rng.random(n) < 0.5 + 0.3 * t).astype(float)
    y = rng.standard_normal(n)
    bad = np.where((t > 0.5) & (s > 0.5))[0][0]
    y[bad] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        lee_bounds(y, t, s)


def test_clean_inputs_still_bound() -> None:
    y, m = _mk()
    r = manski_bounds(y, m)
    assert np.isfinite(r["lb"]) and np.isfinite(r["ub"]) and r["lb"] <= r["ub"]
