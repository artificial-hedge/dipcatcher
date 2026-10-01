"""Tests for ordered choice models (models/ordered_choice.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.ordered_choice import (
    bench_ordered_choice,
    ordered_fit,
    synth_ordered,
)


def _panel(**kw):
    return synth_ordered(seed=28, **kw)


def test_beta_recovery():
    d = _panel(beta=1.0)
    out = ordered_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert abs(float(out["beta_0"]) - 1.0) < 0.2


def test_second_coef_recovery():
    d = _panel()
    out = ordered_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert abs(float(out["beta_1"]) - 0.4) < 0.25


def test_null_beta_small():
    d = _panel(beta=0.0)
    out = ordered_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert abs(float(out["beta_0"])) < 0.3


def test_logit_link():
    d = _panel()
    out = ordered_fit(np.asarray(d["y"]), np.asarray(d["x"]), link="logit")
    # logit sd π/√3 ≈ 1.81 rescales the latent coefficient upward
    assert 1.2 < float(out["beta_0"]) < 2.4


def test_cutpoints_ordered():
    d = _panel()
    out = ordered_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    cuts = [out[f"cut_{j}"] for j in range(int(out["n_categories"]) - 1)]
    assert all(cuts[j] < cuts[j + 1] for j in range(len(cuts) - 1))


def test_validation():
    d = _panel()
    y, x = np.asarray(d["y"]), np.asarray(d["x"])
    with pytest.raises(ValueError):
        ordered_fit(y[:10], x[:10])
    with pytest.raises(ValueError):
        ordered_fit(np.zeros(50), x[:50])  # one category
    with pytest.raises(ValueError):
        ordered_fit(y, x, link="cloglog")
    y2 = y.copy()
    y2[0] = np.nan
    with pytest.raises(ValueError):
        ordered_fit(y2, x)


def test_determinism():
    d = _panel()
    a = ordered_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    b = ordered_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert float(a["beta_0"]) == float(b["beta_0"])


def test_bench_keys():
    out = bench_ordered_choice()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
