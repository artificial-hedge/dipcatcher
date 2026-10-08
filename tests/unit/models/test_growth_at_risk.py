"""Tests for growth_at_risk — ABG vulnerable-growth machinery."""

import numpy as np
import pytest

from quant_fund.models.growth_at_risk import (
    bench_growth_at_risk,
    conditional_quantile_fn,
    growth_at_risk,
    quantile_fit,
    synth_gar,
)


def test_quantile_panel_ordered_at_origin() -> None:
    d = synth_gar(seed=1)
    f = quantile_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    coefs = np.asarray(f["coefs"])
    # at x=0 the fitted quantiles are the intercepts and must rise
    # with tau
    assert np.all(np.diff(coefs[:, 0]) > 0.0)


def test_gar_below_median_and_upside() -> None:
    d = synth_gar(seed=2)
    f = quantile_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    g = growth_at_risk(f, x_new=1.0)
    assert g["gar"] < g["median"] < g["upside"]
    assert g["q_low"] < g["q_high"]


def test_gar_falls_with_conditions() -> None:
    d = synth_gar(seed=3)
    f = quantile_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert growth_at_risk(f, 1.5)["gar"] < growth_at_risk(f, -1.5)["gar"]


def test_downside_asymmetry_negative() -> None:
    d = synth_gar(seed=4)
    f = quantile_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    g = growth_at_risk(f, x_new=1.5)
    assert g["asymmetry"] < 0.0


def test_quantile_fn_monotone() -> None:
    d = synth_gar(seed=5)
    f = quantile_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    q = conditional_quantile_fn(f, 0.3)
    assert np.all(np.diff(q) >= 0.0)


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        quantile_fit(np.ones(10), np.ones(10))
    with pytest.raises(ValueError):
        quantile_fit(np.ones(40), np.ones(39))
    bad = np.ones(50)
    bad[3] = np.nan
    with pytest.raises(ValueError):
        quantile_fit(bad, np.ones(50))
    d = synth_gar(seed=6)
    f = quantile_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    with pytest.raises(ValueError):
        growth_at_risk(f, 0.0, alpha=0.7)


def test_determinism() -> None:
    d = synth_gar(seed=7)
    a = quantile_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    b = quantile_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    np.testing.assert_array_equal(a["coefs"], b["coefs"])


def test_bench_schema_and_score() -> None:
    r = bench_growth_at_risk()
    for k in (
        "synthetic_gar_lo",
        "synthetic_gar_hi",
        "synthetic_gar_drop",
        "synthetic_spread_widen",
        "synthetic_score",
    ):
        assert np.isfinite(r[k])
    assert r["synthetic_score"] == 1.0
