"""Tests for Hansen-Jagannathan SDF diagnostics (metrics/hj_distance.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.hj_distance import (
    bench_hj_distance,
    hj_distance,
    krs_alpha_test,
    sdf_mv_bound,
    sdf_vertex_slope,
    sharpe_of_sdf,
    synth_asset_panel,
)


@pytest.fixture
def good():
    return synth_asset_panel(t=300, n=6, seed=3, misspecified=False)


@pytest.fixture
def bad():
    return synth_asset_panel(t=300, n=6, seed=3, misspecified=True)


def test_synth_shapes(good):
    assert good["R"].shape == (300, 6)
    assert good["F"].shape == (300, 2)


def test_synth_misspecified_drops_factor(bad):
    assert bad["F"].shape == (300, 1)


def test_hj_distance_orders_models(good, bad):
    d_good = hj_distance(good["R"], good["F"])
    d_bad = hj_distance(bad["R"], bad["F"])
    assert d_good < d_bad
    assert d_good < 0.2
    assert d_bad > 0.1


def test_hj_distance_weightings(good):
    d1 = hj_distance(good["R"], good["F"], weighting="returns")
    d2 = hj_distance(good["R"], good["F"], weighting="second")
    assert d1 >= 0 and d2 >= 0
    assert math.isfinite(d1) and math.isfinite(d2)


def test_hj_distance_deterministic(good):
    assert hj_distance(good["R"], good["F"]) == hj_distance(good["R"], good["F"])


def test_krs_alpha_p_orders(good, bad):
    p_good = krs_alpha_test(good["R"], good["F"])["krs_p"]
    p_bad = krs_alpha_test(bad["R"], bad["F"])["krs_p"]
    assert p_good > p_bad
    assert p_good > 0.1


def test_sdf_bound_shape(good):
    b = sdf_mv_bound(good["R"], n_points=25)
    assert b.shape == (25, 2)
    assert np.all(np.isfinite(b))
    assert np.all(b[:, 1] >= 0)


def test_vertex_slope_matches_bound(good):
    b = sdf_mv_bound(good["R"], n_points=41)
    vertex = sdf_vertex_slope(good["R"])
    ratio = b[:, 1] / np.maximum(b[:, 0], 1e-9)
    assert np.min(ratio) == pytest.approx(vertex, rel=0.1)


def test_sharpe_positive(good):
    assert sharpe_of_sdf(good["R"]) > 0


def test_hj_input_validation():
    with pytest.raises(ValueError):
        hj_distance(np.zeros((10, 3)), np.zeros((10, 1)))
    with pytest.raises(ValueError):
        hj_distance(np.zeros((30, 3)), np.zeros((10, 1)))
    with pytest.raises(ValueError):
        hj_distance(np.zeros((30, 3)), np.zeros((30, 1)), weighting="bad")


def test_bench_keys():
    out = bench_hj_distance()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_hj_true"] < 0.2
    assert out["synthetic_hj_misspec"] > out["synthetic_hj_true"]
    assert out["synthetic_hj_margin"] > 0.1
    assert out["synthetic_krs_p_true"] > 0.05
    assert out["synthetic_krs_p_misspec"] < 0.05
    assert out["synthetic_bound_covers_sr"] == 1.0
    assert out["synthetic_determinism"] == 1.0
