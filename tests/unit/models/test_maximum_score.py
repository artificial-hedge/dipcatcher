"""Tests for Manski maximum score (models/maximum_score.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.maximum_score import (
    bench_maximum_score,
    maximum_score,
    synth_binary_response,
)


def test_direction_recovered():
    d = synth_binary_response(beta=(0.8, -0.5), seed=37)
    out = maximum_score(np.asarray(d["y"]), np.asarray(d["x"]), seed=37)
    bt = np.array([0.8, -0.5]) / np.linalg.norm([0.8, -0.5])
    bh = np.array([float(out["beta_1"]), float(out["beta_2"])])
    bh = bh / np.linalg.norm(bh)
    assert abs(float(bh @ bt)) > 0.9


def test_score_beats_chance():
    d = synth_binary_response(beta=(0.8, -0.5), seed=37)
    out = maximum_score(np.asarray(d["y"]), np.asarray(d["x"]), seed=37)
    assert float(out["score_rate"]) > 0.6


def test_null_no_signal():
    d = synth_binary_response(beta=(0.0, 0.0), seed=37)
    out = maximum_score(np.asarray(d["y"]), np.asarray(d["x"]), seed=37)
    assert float(out["score_rate"]) < 0.62


def test_heteroskedastic_robust():
    d = synth_binary_response(beta=(0.8, -0.5), hetero=True, seed=37)
    out = maximum_score(np.asarray(d["y"]), np.asarray(d["x"]), seed=37)
    bt = np.array([0.8, -0.5]) / np.linalg.norm([0.8, -0.5])
    bh = np.array([float(out["beta_1"]), float(out["beta_2"])])
    bh = bh / np.linalg.norm(bh)
    assert abs(float(bh @ bt)) > 0.85


def test_beta_unit_norm():
    d = synth_binary_response(seed=37)
    out = maximum_score(np.asarray(d["y"]), np.asarray(d["x"]), seed=37)
    b = np.array([float(out[f"beta_{j}"]) for j in range(3)])
    assert abs(float(np.linalg.norm(b)) - 1.0) < 1e-9


def test_validation():
    d = synth_binary_response(seed=37)
    y, x = np.asarray(d["y"]), np.asarray(d["x"])
    with pytest.raises(ValueError):
        maximum_score(y[:30], x[:30])
    with pytest.raises(ValueError):
        maximum_score(np.full(60, 1.0), x[:60])
    y2 = y.copy()
    y2[0] = 2.0
    with pytest.raises(ValueError):
        maximum_score(y2, x)
    with pytest.raises(ValueError):
        maximum_score(y, np.column_stack([x, x[:, 0]]))


def test_determinism():
    d = synth_binary_response(seed=37)
    a = maximum_score(np.asarray(d["y"]), np.asarray(d["x"]), seed=37)
    b = maximum_score(np.asarray(d["y"]), np.asarray(d["x"]), seed=37)
    assert float(a["beta_1"]) == float(b["beta_1"])


def test_bench_keys():
    out = bench_maximum_score()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
