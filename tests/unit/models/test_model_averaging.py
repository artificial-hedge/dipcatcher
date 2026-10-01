"""Tests for Mallows model averaging (models/model_averaging.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.model_averaging import (
    bench_model_averaging,
    mma_fit,
    synth_nested,
)


def test_weights_track_complexity():
    d = synth_nested(decay=0.45, seed=37)
    d_sharp = synth_nested(decay=0.01, seed=37)
    out = mma_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    out_s = mma_fit(np.asarray(d_sharp["y"]), np.asarray(d_sharp["x"]))
    assert float(out["w_argmax_model"]) > float(out_s["w_argmax_model"])


def test_sharp_concentrates():
    d = synth_nested(decay=0.01, seed=37)
    out = mma_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert float(out["w_argmax_model"]) <= 3.0


def test_mspe_competitive():
    d = synth_nested(decay=0.5, seed=37)
    out = mma_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert float(out["mspe_mma"]) <= float(out["mspe_single"]) * 1.05


def test_weights_simplex():
    d = synth_nested(seed=37)
    out = mma_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert 0.0 <= float(out["w_top"]) <= 1.0
    assert float(out["w_entropy"]) >= 0.0


def test_validation():
    d = synth_nested(seed=37)
    y, x = np.asarray(d["y"]), np.asarray(d["x"])
    with pytest.raises(ValueError):
        mma_fit(y[:40], x[:40])
    with pytest.raises(ValueError):
        mma_fit(y, x[:, :1])  # k<2
    with pytest.raises(ValueError):
        mma_fit(y, np.ones((y.size, 2)))  # constant regressors
    with pytest.raises(ValueError):
        mma_fit(y * 0.0, x)  # constant y
    bad = y.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        mma_fit(bad, x)


def test_determinism():
    d = synth_nested(seed=37)
    a = mma_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    b = mma_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert float(a["w_top"]) == float(b["w_top"])
    assert float(a["mspe_mma"]) == float(b["mspe_mma"])


def test_bench_keys():
    out = bench_model_averaging()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
