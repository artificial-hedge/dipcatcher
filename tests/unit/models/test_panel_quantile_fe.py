"""Tests for panel quantile FE via moments (models/panel_quantile_fe.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.panel_quantile_fe import (
    bench_panel_quantile_fe,
    panel_quantile_fe,
    quantile_gradient_test,
    synth_locscale_panel,
)


def _panel(**kw):
    return synth_locscale_panel(seed=20, **kw)


def test_median_recovers_beta():
    d = _panel()
    out = panel_quantile_fe(
        np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["groups"]), tau=0.5
    )
    assert abs(float(np.asarray(out["beta_tau"])[0]) - 1.0) < 0.3


def test_gradient_positive_under_hetero():
    d = _panel(hetero_sd=0.6)
    grad = quantile_gradient_test(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["groups"]))
    betas = np.asarray(grad["betas"])[:, 0]
    assert betas[-1] > betas[0] + 0.3


def test_tau_monotone():
    d = _panel(hetero_sd=0.6)
    grad = quantile_gradient_test(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["groups"]))
    betas = np.asarray(grad["betas"])[:, 0]
    assert betas[4] > betas[2] > betas[0] - 0.4


def test_gradient_shape():
    d = _panel()
    grad = quantile_gradient_test(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["groups"]))
    assert np.asarray(grad["betas"]).shape == (5, 1)


def test_group_demeaning_removes_fe():
    d = _panel()
    # corrupt: add huge group effects — within estimator must ignore them
    y = np.asarray(d["y"]).copy()
    ga = np.asarray(d["groups"])
    y += ga * 100.0
    out = panel_quantile_fe(y, np.asarray(d["x"]), ga, tau=0.5)
    assert abs(float(np.asarray(out["beta_tau"])[0]) - 1.0) < 0.3


def test_validation():
    d = _panel()
    y = np.asarray(d["y"])
    x = np.asarray(d["x"])
    g = np.asarray(d["groups"])
    with pytest.raises(ValueError):
        panel_quantile_fe(y[:5], x, g)
    with pytest.raises(ValueError):
        panel_quantile_fe(y, x[:4], g)
    with pytest.raises(ValueError):
        panel_quantile_fe(y, x, g, tau=1.2)
    with pytest.raises(ValueError):
        panel_quantile_fe(y, x, np.zeros(y.size))
    x2 = x.copy()
    x2[0, 0] = np.nan
    with pytest.raises(ValueError):
        panel_quantile_fe(y, x2, g)


def test_determinism():
    d = _panel()
    a = panel_quantile_fe(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["groups"]), tau=0.75)
    b = panel_quantile_fe(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["groups"]), tau=0.75)
    assert np.array_equal(np.asarray(a["beta_tau"]), np.asarray(b["beta_tau"]))


def test_bench_keys():
    out = bench_panel_quantile_fe()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects_gradient"] == 1.0
    assert out["synthetic_determinism"] == 1.0
