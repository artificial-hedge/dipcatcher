"""Tests for nardl — asymmetric ARDL cointegration."""

import numpy as np
import pytest

from quant_fund.models.nardl import (
    asymmetric_multipliers,
    bench_nardl,
    bounds_f,
    fit_nardl,
    partial_sums,
    synth_nardl,
    wald_symmetry,
)


def test_partial_sums_shape_and_content() -> None:
    dx = np.array([1.0, -2.0, 0.5, -0.5, 2.0])
    xp, xm = partial_sums(dx)
    assert xp.shape == xm.shape == (6,)
    np.testing.assert_allclose(xp, [0, 1, 1, 1.5, 1.5, 3.5])
    np.testing.assert_allclose(xm, [0, 0, -2, -2, -2.5, -2.5])


def test_recovers_long_run_arms() -> None:
    d = synth_nardl(seed=1)
    f = fit_nardl(np.asarray(d["y"]), np.asarray(d["x"]))
    assert abs(float(f["l_plus"]) - float(d["l_plus"])) < 0.15
    assert abs(float(f["l_minus"]) - float(d["l_minus"])) < 0.15
    assert float(f["rho"]) < 0.0


def test_symmetry_rejected_when_asymmetric() -> None:
    d = synth_nardl(seed=2)
    f = fit_nardl(np.asarray(d["y"]), np.asarray(d["x"]))
    w = wald_symmetry(f)
    assert w["p_value"] < 0.05


def test_symmetry_retained_when_symmetric() -> None:
    d = synth_nardl(seed=3, l_plus=0.5, l_minus=0.5)
    f = fit_nardl(np.asarray(d["y"]), np.asarray(d["x"]))
    w = wald_symmetry(f)
    assert w["p_value"] > 0.01


def test_bounds_test_significant() -> None:
    d = synth_nardl(seed=4)
    f = fit_nardl(np.asarray(d["y"]), np.asarray(d["x"]))
    b = bounds_f(f, np.asarray(d["y"]), np.asarray(d["x"]))
    assert b["f_stat"] > 3.0
    assert b["df1"] == 3.0


def test_multipliers_converge_to_long_run() -> None:
    d = synth_nardl(seed=5)
    f = fit_nardl(np.asarray(d["y"]), np.asarray(d["x"]))
    m = asymmetric_multipliers(f, np.asarray(d["y"])[-5:], 0.0, 0.0, h=30)
    assert m["m_plus"][-1] == pytest.approx(float(f["l_plus"]), abs=0.15)
    assert m["m_minus"][-1] == pytest.approx(float(f["l_minus"]), abs=0.15)


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        fit_nardl(np.ones(10), np.ones(10))
    with pytest.raises(ValueError):
        fit_nardl(np.ones(40), np.ones(39))
    bad = np.ones(60)
    bad[5] = np.nan
    with pytest.raises(ValueError):
        fit_nardl(bad, np.ones(60))
    with pytest.raises(ValueError):
        fit_nardl(np.ones(60), np.ones(60), p=0)
    with pytest.raises(ValueError):
        partial_sums(np.array([1.0, np.nan, 2.0, 3.0]))


def test_determinism() -> None:
    d = synth_nardl(seed=8)
    a = fit_nardl(np.asarray(d["y"]), np.asarray(d["x"]))
    b = fit_nardl(np.asarray(d["y"]), np.asarray(d["x"]))
    np.testing.assert_array_equal(a["beta"], b["beta"])


def test_bench_schema_and_score() -> None:
    r = bench_nardl(seed=3)
    for k in ("l_plus", "l_minus", "lp_err", "lm_err", "wald_sym", "f_bounds", "score"):
        assert np.isfinite(r[k])
    assert r["score"] == 1.0
