"""Tests for mixed logit simulated MLE (models/mixed_logit.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.mixed_logit import (
    bench_mixed_logit,
    mixed_logit_fit,
    synth_mixed_choice,
)


def test_mu_recovered():
    d = synth_mixed_choice(mu=1.0, sigma=1.0, seed=37)
    out = mixed_logit_fit(np.asarray(d["choice"]), np.asarray(d["x"]), seed=37)
    assert abs(float(out["mu"]) - 1.0) < 0.4


def test_sigma_detected():
    d = synth_mixed_choice(mu=1.0, sigma=1.2, seed=37)
    out = mixed_logit_fit(np.asarray(d["choice"]), np.asarray(d["x"]), seed=37)
    assert float(out["sigma"]) > 0.5


def test_homogeneous_sigma_small():
    d = synth_mixed_choice(mu=1.0, sigma=0.01, seed=37)
    out = mixed_logit_fit(np.asarray(d["choice"]), np.asarray(d["x"]), seed=37)
    assert float(out["sigma"]) < 0.4


def test_hetero_improves_fit():
    d = synth_mixed_choice(mu=1.0, sigma=1.5, seed=37)
    out = mixed_logit_fit(np.asarray(d["choice"]), np.asarray(d["x"]), seed=37)
    assert float(out["ll_mixed"]) > float(out["ll_mnl"])


def test_validation():
    d = synth_mixed_choice(seed=37)
    c, x = np.asarray(d["choice"]), np.asarray(d["x"])
    with pytest.raises(ValueError):
        mixed_logit_fit(c[:30], x[:30])
    with pytest.raises(ValueError):
        mixed_logit_fit(c, x[:, :, :1] * 0.0)
    with pytest.raises(ValueError):
        mixed_logit_fit(c, x[:, :2, :])  # <3 alternatives
    c2 = c.copy()
    c2[0] = np.nan
    with pytest.raises(ValueError):
        mixed_logit_fit(c2, x)
    with pytest.raises(ValueError):
        mixed_logit_fit(c, x, n_draws=10)


def test_determinism():
    d = synth_mixed_choice(seed=37)
    a = mixed_logit_fit(np.asarray(d["choice"]), np.asarray(d["x"]), seed=37)
    b = mixed_logit_fit(np.asarray(d["choice"]), np.asarray(d["x"]), seed=37)
    assert float(a["mu"]) == float(b["mu"])
    assert float(a["sigma"]) == float(b["sigma"])


def test_bench_keys():
    out = bench_mixed_logit()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
