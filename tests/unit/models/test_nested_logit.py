"""Tests for nested logit FIML (models/nested_logit.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.nested_logit import (
    bench_nested_logit,
    nested_logit_fit,
    synth_nested_choice,
)


def test_beta_recovered():
    d = synth_nested_choice(beta=1.2, rho=0.7, seed=37)
    out = nested_logit_fit(np.asarray(d["choice"]), np.asarray(d["x"]), np.asarray(d["nest"]))
    assert abs(float(out["beta_0"]) - 1.2) < 0.4


def test_lambda_below_one():
    d = synth_nested_choice(beta=1.0, rho=0.7, seed=37)
    out = nested_logit_fit(np.asarray(d["choice"]), np.asarray(d["x"]), np.asarray(d["nest"]))
    assert float(out["lambda"]) < 0.98


def test_iid_lambda_near_one():
    d = synth_nested_choice(beta=1.0, rho=0.05, seed=37)
    out = nested_logit_fit(np.asarray(d["choice"]), np.asarray(d["x"]), np.asarray(d["nest"]))
    assert float(out["lambda"]) > 0.5


def test_null_beta():
    d = synth_nested_choice(beta=0.0, seed=37)
    out = nested_logit_fit(np.asarray(d["choice"]), np.asarray(d["x"]), np.asarray(d["nest"]))
    assert abs(float(out["beta_0"])) < 0.35


def test_validation():
    d = synth_nested_choice(seed=37)
    c, x, ns = np.asarray(d["choice"]), np.asarray(d["x"]), np.asarray(d["nest"])
    with pytest.raises(ValueError):
        nested_logit_fit(c, x[:, :, 0], ns)  # x not 3-d
    with pytest.raises(ValueError):
        nested_logit_fit(c[:30], x[:30], ns)
    with pytest.raises(ValueError):
        nested_logit_fit(c, x, ns[:2])  # wrong nest length
    c2 = c.copy()
    c2[0] = 99
    with pytest.raises(ValueError):
        nested_logit_fit(c2, x, ns)


def test_determinism():
    d = synth_nested_choice(seed=37)
    a = nested_logit_fit(np.asarray(d["choice"]), np.asarray(d["x"]), np.asarray(d["nest"]))
    b = nested_logit_fit(np.asarray(d["choice"]), np.asarray(d["x"]), np.asarray(d["nest"]))
    assert float(a["beta_0"]) == float(b["beta_0"])
    assert float(a["lambda"]) == float(b["lambda"])


def test_bench_keys():
    out = bench_nested_logit()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
