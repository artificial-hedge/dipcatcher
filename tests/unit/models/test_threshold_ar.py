"""Tests for SETAR threshold autoregression (models/threshold_ar.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.threshold_ar import (
    bench_threshold_ar,
    setar_fit,
    synth_setar,
)


def test_threshold_recovered():
    out = setar_fit(synth_setar(seed=37))
    assert abs(float(out["c_hat"]) - 0.5) < 0.25


def test_regime_phis():
    out = setar_fit(synth_setar(seed=37, phi_lo=0.85, phi_hi=0.3))
    assert float(out["phi_lo"]) > float(out["phi_hi"]) + 0.3


def test_linearity_f():
    out = setar_fit(synth_setar(seed=37))
    assert float(out["f_linearity"]) > 2.0


def test_linear_series_small_gap():
    rng = np.random.default_rng(37)
    y = np.zeros(500)
    for t in range(1, 500):
        y[t] = 0.6 * y[t - 1] + rng.normal(0.0, 0.2)
    out = setar_fit(y)
    assert abs(float(out["phi_lo"]) - float(out["phi_hi"])) < 0.5


def test_sse_beats_linear():
    out = setar_fit(synth_setar(seed=37))
    assert float(out["sse_tar"]) < float(out["sse_linear"])


def test_validation():
    y = synth_setar(seed=37)
    with pytest.raises(ValueError):
        setar_fit(y[:30])
    with pytest.raises(ValueError):
        setar_fit(y, delay=0)
    with pytest.raises(ValueError):
        setar_fit(y, delay=1, trim=0.5)
    y2 = y.copy()
    y2[10] = np.nan
    with pytest.raises(ValueError):
        setar_fit(y2)


def test_determinism():
    a, b = setar_fit(synth_setar(seed=37)), setar_fit(synth_setar(seed=37))
    assert float(a["c_hat"]) == float(b["c_hat"])


def test_bench_keys():
    out = bench_threshold_ar()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
