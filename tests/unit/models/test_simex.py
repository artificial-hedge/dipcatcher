"""Tests for SIMEX measurement-error correction (models/simex.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.simex import bench_simex, simex, synth_mismeasured


def _panel(**kw):
    return synth_mismeasured(seed=32, **kw)


def _fit(d):
    return simex(
        np.asarray(d["y"]),
        np.asarray(d["w"]),
        float(np.asarray(d["sigma_u"])[0]),
        seed=32,
    )


def test_corrected_beats_naive():
    out = _fit(_panel(beta=1.0, reliability=0.6))
    assert abs(float(out["corrected"]) - 1.0) < abs(float(out["naive"]) - 1.0)


def test_naive_shows_attenuation():
    out = _fit(_panel(beta=1.0, reliability=0.6))
    assert float(out["naive"]) < 0.8  # reliability .6 → ~.6 expected


def test_mild_error_less_bias():
    hi = _fit(_panel(beta=1.0, reliability=0.9))
    lo = _fit(_panel(beta=1.0, reliability=0.5))
    assert abs(float(hi["corrected"]) - 1.0) < abs(float(lo["corrected"]) - 1.0) + 0.15


def test_null_beta_small():
    out = _fit(_panel(beta=0.0, reliability=0.6))
    assert abs(float(out["corrected"])) < 0.35


def test_extrapolation_fit():
    out = _fit(_panel())
    assert float(out["extrapolation_r2"]) > 0.8


def test_validation():
    d = _panel()
    y, w = np.asarray(d["y"]), np.asarray(d["w"])
    su = float(np.asarray(d["sigma_u"])[0])
    with pytest.raises(ValueError):
        simex(y[:20], w[:20], su)
    with pytest.raises(ValueError):
        simex(y, w, -1.0)
    with pytest.raises(ValueError):
        simex(y, w, su, lambdas=np.array([1.0]))
    with pytest.raises(ValueError):
        synth_mismeasured(reliability=1.5)
    y2 = y.copy()
    y2[0] = np.nan
    with pytest.raises(ValueError):
        simex(y2, w, su)


def test_determinism():
    a, b = _fit(_panel()), _fit(_panel())
    assert float(a["corrected"]) == float(b["corrected"])


def test_bench_keys():
    out = bench_simex()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_beats_naive"] == 1.0
