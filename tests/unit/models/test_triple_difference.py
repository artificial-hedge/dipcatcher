"""Tests for triple difference estimation (models/triple_difference.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.triple_difference import (
    bench_triple_difference,
    synth_ddd,
    triple_difference,
)


def _panel(**kw):
    return synth_ddd(seed=29, **kw)


def _fit(d):
    return triple_difference(
        np.asarray(d["y"]),
        np.asarray(d["post"]),
        np.asarray(d["treat"]),
        np.asarray(d["elig"]),
    )


def test_ddd_recovers_effect():
    out = _fit(_panel(effect=1.0))
    assert abs(float(out["ddd"]) - 1.0) < 0.2


def test_ddd_corrects_confounded_did():
    out = _fit(_panel(effect=1.0, cell_confound=0.8))
    assert abs(float(out["ddd"]) - 1.0) < 0.25


def test_cell_contrast_matches():
    out = _fit(_panel())
    assert abs(float(out["ddd"]) - float(out["cell_contrast"])) < 1e-8


def test_null_ddd_small():
    out = _fit(_panel(effect=0.0))
    assert abs(float(out["ddd"])) < 0.2
    assert float(out["p_value"]) > 0.01


def test_significance_on_effect():
    out = _fit(_panel(effect=1.0))
    assert float(out["p_value"]) < 0.01


def test_validation():
    d = _panel()
    y = np.asarray(d["y"])
    p, t, e = (
        np.asarray(d["post"]),
        np.asarray(d["treat"]),
        np.asarray(d["elig"]),
    )
    with pytest.raises(ValueError):
        triple_difference(y[:20], p[:20], t[:20], e[:20])
    with pytest.raises(ValueError):
        triple_difference(y, p, t, e[:10])
    t2 = t.copy()
    t2[0] = 2.0
    with pytest.raises(ValueError):
        triple_difference(y, p, t2, e)
    y2 = y.copy()
    y2[0] = np.nan
    with pytest.raises(ValueError):
        triple_difference(y2, p, t, e)


def test_determinism():
    a, b = _fit(_panel()), _fit(_panel())
    assert float(a["ddd"]) == float(b["ddd"])


def test_bench_keys():
    out = bench_triple_difference()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_beats_naive"] == 1.0
