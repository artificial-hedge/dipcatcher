"""Tests for specification-curve analysis (models/specification_curve.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.specification_curve import (
    bench_specification_curve,
    spec_curve_shuffle_p,
    spec_grid,
    specification_curve,
    synth_multiverse,
)


def _panel(**kw):
    kw.setdefault("effect", 0.5)
    return synth_multiverse(seed=18, **kw)


def test_grid_nonempty():
    g = spec_grid()
    assert len(g) > 3
    assert all(len(s) == 3 for s in g)


def test_median_recovers_effect():
    d = _panel()
    sc = specification_curve(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["covariate"]))
    assert abs(float(sc["median"]) - 0.5) < 0.3
    assert float(sc["n_specs"]) >= 5.0


def test_sorted_curve():
    d = _panel()
    sc = specification_curve(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["covariate"]))
    es = np.asarray(sc["effects_sorted"])
    assert np.all(np.diff(es) >= 0.0)


def test_shuffle_p_low_for_real():
    d = _panel()
    sh = spec_curve_shuffle_p(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["covariate"]),
        n_shuffles=50,
        seed=1,
    )
    assert sh["p_value"] < 0.3


def test_shuffle_p_high_for_null():
    d = _panel(effect=0.0)
    sh = spec_curve_shuffle_p(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["covariate"]),
        n_shuffles=50,
        seed=2,
    )
    assert sh["p_value"] > 0.3


def test_labels_match_specs():
    d = _panel()
    sc = specification_curve(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["covariate"]))
    assert len(sc["labels"]) == int(sc["n_specs"])


def test_validation():
    d = _panel()
    y = np.asarray(d["y"])
    t = np.asarray(d["treat"])
    c = np.asarray(d["covariate"])
    with pytest.raises(ValueError):
        specification_curve(y[:5], t, c)
    with pytest.raises(ValueError):
        specification_curve(y, t, c[:3])
    y2 = y.copy()
    y2[0] = np.nan
    with pytest.raises(ValueError):
        specification_curve(y2, t, c)
    with pytest.raises(ValueError):
        specification_curve(y, t, c, grid=[])


def test_determinism():
    d = _panel()
    a = specification_curve(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["covariate"]))
    b = specification_curve(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["covariate"]))
    assert float(a["median"]) == float(b["median"])


def test_bench_keys():
    out = bench_specification_curve()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
