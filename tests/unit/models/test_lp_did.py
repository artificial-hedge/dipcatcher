"""Tests for LP-DiD estimation (models/lp_did.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.lp_did import bench_lp_did, lp_did, synth_lp_did


def _panel(**kw):
    return synth_lp_did(seed=33, **kw)


def _fit(d):
    return lp_did(
        np.asarray(d["y"]),
        np.asarray(d["unit"]),
        np.asarray(d["time"]),
        np.asarray(d["treat_time"]),
    )


def test_ramp_tracked():
    out = _fit(_panel(effect_growth=0.4))
    for h, expect in ((0, 0.4), (1, 0.8), (2, 1.2)):
        v = float(out[f"eff_h{h}"])
        assert abs(v - expect) < 0.3


def test_growing_effects():
    out = _fit(_panel(effect_growth=0.4))
    assert float(out["eff_h2"]) > float(out["eff_h0"])


def test_null_flat():
    out = _fit(_panel(effect_growth=0.0))
    for h in (0, 1, 2):
        v = out.get(f"eff_h{h}", math.nan)
        if math.isfinite(v):
            assert abs(v) < 0.4


def test_significance():
    out = _fit(_panel(effect_growth=0.4))
    assert float(out["max_abs_z"]) > 3.0


def test_horizon_count():
    out = _fit(_panel())
    assert "eff_h0" in out and "se_h0" in out and "p_h0" in out


def test_validation():
    d = _panel()
    y, u, t, g = (
        np.asarray(d["y"]),
        np.asarray(d["unit"]),
        np.asarray(d["time"]),
        np.asarray(d["treat_time"]),
    )
    with pytest.raises(ValueError):
        lp_did(y[:20], u[:20], t[:20], g[:20])
    with pytest.raises(ValueError):
        lp_did(y, u, t, np.full(g.size, np.nan))
    with pytest.raises(ValueError):
        lp_did(y, u[:10], t, g)
    y2 = y.copy()
    y2[0] = np.nan
    with pytest.raises(ValueError):
        lp_did(y2, u, t, g)


def test_determinism():
    a, b = _fit(_panel()), _fit(_panel())
    assert float(a["eff_h0"]) == float(b["eff_h0"])


def test_bench_keys():
    out = bench_lp_did()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
