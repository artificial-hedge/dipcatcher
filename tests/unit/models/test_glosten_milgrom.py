"""Tests for glosten_milgrom — sequential-trade price discovery."""

import numpy as np
import pytest

from quant_fund.models.glosten_milgrom import (
    bench_glosten_milgrom,
    gm_quotes,
    gm_update,
    lb_p,
    synth_gm,
)


def test_quotes_bracket_mid() -> None:
    qq = gm_quotes(0.5, 1.2, 0.8, 0.3)
    assert qq["bid"] < 1.0 < qq["ask"]
    assert qq["spread"] > 0
    # symmetric prior -> symmetric quotes
    assert abs((qq["ask"] + qq["bid"]) / 2 - 1.0) < 1e-12


def test_spread_vanishes_without_informed() -> None:
    qq = gm_quotes(0.5, 1.2, 0.8, 0.001)
    assert qq["spread"] < 0.01


def test_update_directions() -> None:
    assert gm_update(0.5, 1, 0.3) > 0.5
    assert gm_update(0.5, -1, 0.3) < 0.5
    # buy+sell roundtrip ~ preserves q
    q = gm_update(gm_update(0.5, 1, 0.3), -1, 0.3)
    assert abs(q - 0.5) < 0.05


def test_posterior_learns() -> None:
    _, _, _, _, qpath, v_high = synth_gm(seed=1)
    acc = np.mean((qpath[400:] > 0.5) == (v_high > 0.5))
    assert acc > 0.9


def test_naive_bounce_negative() -> None:
    _, _, naive_inc, _, _, _ = synth_gm(seed=2)
    r = float(np.corrcoef(naive_inc[1:], naive_inc[:-1])[0, 1])
    assert r < -0.2


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        gm_quotes(0.0, 1.2, 0.8, 0.3)
    with pytest.raises(ValueError):
        gm_quotes(0.5, 0.8, 1.2, 0.3)
    with pytest.raises(ValueError):
        gm_quotes(0.5, 1.2, 0.8, 1.5)
    with pytest.raises(ValueError):
        gm_update(0.5, 0, 0.3)
    with pytest.raises(ValueError):
        gm_update(1.5, 1, 0.3)


def test_determinism() -> None:
    a = synth_gm(seed=7)
    b = synth_gm(seed=7)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_lb_p_uniform_on_white() -> None:
    rng = np.random.default_rng(0)
    p = lb_p(rng.normal(size=500))
    assert p > 0.01


def test_bench_schema_and_score() -> None:
    r = bench_glosten_milgrom()
    for k in (
        "synthetic_posterior_acc",
        "synthetic_theory_spread",
        "synthetic_first_spread",
        "synthetic_late_spread",
        "synthetic_gm_lag1_acf",
        "synthetic_naive_lag1_acf",
        "synthetic_score",
    ):
        assert np.isfinite(r[k])
    assert r["synthetic_score"] == 1.0
