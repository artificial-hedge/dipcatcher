"""Tests for censored quantile regression (models/censored_quantile.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.censored_quantile import (
    bench_censored_quantile,
    censored_quantile,
    synth_censored,
)


def _panel(**kw):
    return synth_censored(seed=36, **kw)


def _fit(d, c):
    return censored_quantile(np.asarray(d["y"]), np.asarray(d["x"]), censor_at=c)


def test_slope_recovery():
    out = _fit(_panel(beta=1.0, censor_at=1.5), 1.5)
    assert abs(float(out["beta_1"]) - 1.0) < 0.2


def test_beats_naive():
    out = _fit(_panel(beta=1.0, censor_at=1.5), 1.5)
    assert abs(float(out["beta_1"]) - 1.0) < abs(float(out["beta_naive_1"]) - 1.0)


def test_censor_share_detected():
    out = _fit(_panel(beta=1.0, censor_at=1.5), 1.5)
    assert 0.05 < float(out["censor_share"]) < 0.4


def test_null_beta():
    out = _fit(_panel(beta=0.0, censor_at=0.4), 0.4)
    assert abs(float(out["beta_1"])) < 0.3


def test_heavy_censoring():
    out = _fit(_panel(beta=1.0, censor_at=0.5), 0.5)
    assert abs(float(out["beta_1"]) - 1.0) < 0.35


def test_validation():
    d = _panel()
    y, x = np.asarray(d["y"]), np.asarray(d["x"])
    with pytest.raises(ValueError):
        censored_quantile(y[:10], x[:10], 1.0)
    with pytest.raises(ValueError):
        censored_quantile(y, x, 1.0, tau=0.99)
    with pytest.raises(ValueError):
        censored_quantile(y, x, -100.0)  # censors everything
    with pytest.raises(ValueError):
        censored_quantile(y, x, 100.0)  # censors nothing
    y2 = y.copy()
    y2[0] = np.nan
    with pytest.raises(ValueError):
        censored_quantile(y2, x, 1.5)


def test_determinism():
    a, b = _fit(_panel(), 1.5), _fit(_panel(), 1.5)
    assert float(a["beta_1"]) == float(b["beta_1"])


def test_bench_keys():
    out = bench_censored_quantile()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_beats_naive"] == 1.0
