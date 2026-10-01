"""Tests for extreme-value tail estimation (metrics/extreme_value.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.extreme_value import (
    bench_extreme_value,
    gpd_fit,
    hill_index,
    hill_plot,
    mean_excess,
    return_level,
    synth_frechet,
    tail_risk,
)


@pytest.fixture
def heavy():
    return synth_frechet(n=4000, alpha=3.0, seed=7)


def test_hill_recovers_alpha(heavy):
    out = hill_index(np.asarray(heavy["x"]), k=200)
    assert abs(out["alpha"] - 3.0) < 0.6
    assert out["se"] > 0
    assert out["k"] >= 10


def test_hill_plot_stable(heavy):
    hp = hill_plot(np.asarray(heavy["x"]))
    xi = np.asarray(hp["xi"])
    tail = xi[np.asarray(hp["k"]) > 60]
    assert float(np.std(tail)) < 0.3


def test_gpd_fit():
    rng = np.random.default_rng(0)
    y = rng.standard_gamma(2.0, 800)  # light-tail exceedances
    fit = gpd_fit(y)
    assert fit["beta"] > 0
    assert -0.5 < fit["xi"] < 1.5
    assert fit["n_exc"] == 800


def test_tail_risk_es_above_var(heavy):
    tr = tail_risk(np.asarray(heavy["x"]))
    assert tr["es"] > tr["var"] > tr["u"]
    assert 0 < tr["xi"] < 1


def test_return_level_grows(heavy):
    r100 = return_level(np.asarray(heavy["x"]), m=100)
    r400 = return_level(np.asarray(heavy["x"]), m=400)
    assert r400["return_level"] > r100["return_level"] > 0


def test_mean_excess_finite(heavy):
    me = mean_excess(np.asarray(heavy["x"]))
    assert np.isfinite(np.asarray(me["e"])).all()
    assert (np.asarray(me["e"]) > 0).all()


def test_thin_tail_flag():
    rng = np.random.default_rng(1)
    x = np.abs(rng.standard_normal(500)) * 0.01 + 1.0  # bounded-ish tail
    out = hill_index(x)
    assert out["alpha"] > 3.0  # light tail → larger alpha


def test_validation():
    with pytest.raises(ValueError):
        hill_index(np.ones(10))
    with pytest.raises(ValueError):
        hill_index(np.arange(50.0), k=1)
    with pytest.raises(ValueError):
        gpd_fit(np.ones(5))
    with pytest.raises(ValueError):
        tail_risk(np.ones(100), u_quantile=0.9, p_exceed=0.9)


def test_determinism(heavy):
    a = hill_index(np.asarray(heavy["x"]))["alpha"]
    b = hill_index(np.asarray(heavy["x"]))["alpha"]
    assert a == b


def test_bench_keys():
    out = bench_extreme_value()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_hill_alpha_err"] < 0.5
    assert out["synthetic_es_gt_var"] == 1.0
    assert out["synthetic_determinism"] == 1.0
