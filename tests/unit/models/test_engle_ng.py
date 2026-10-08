"""Tests for engle_ng — sign/size-bias asymmetry diagnostics."""

import numpy as np
import pytest

from quant_fund.models.engle_ng import (
    _garch11_filter,
    bench_engle_ng,
    engle_ng_test,
    synth_engle_ng,
)


def test_leverage_detected_under_symmetric_fit() -> None:
    lev, _ = synth_engle_ng(seed=1)
    r = engle_ng_test(lev / _garch11_filter(lev))
    assert r["p_joint"] < 0.01


def test_symmetric_passes_under_fit() -> None:
    _, sym = synth_engle_ng(seed=2)
    r = engle_ng_test(sym / _garch11_filter(sym))
    assert r["p_joint"] > 0.01


def test_sign_coefficient_nonzero_on_leverage() -> None:
    lev, _ = synth_engle_ng(seed=3)
    r = engle_ng_test(lev / _garch11_filter(lev))
    assert r["b_sign"] < 0 or r["t_neg_size"] > 0


def test_garch_filter_tracks_vol() -> None:
    _, sym = synth_engle_ng(seed=4)
    sig = _garch11_filter(sym)
    assert np.all(sig > 0)
    z = sym / sig
    assert 0.5 < np.std(z) < 2.5


def test_fail_closed() -> None:
    lev, _ = synth_engle_ng(seed=5)
    with pytest.raises(ValueError):
        engle_ng_test(lev[:100])
    with pytest.raises(ValueError):
        engle_ng_test(np.full(500, np.nan))
    with pytest.raises(ValueError):
        engle_ng_test(np.ones(500))


def test_determinism() -> None:
    lev, _ = synth_engle_ng(seed=6)
    z = lev / _garch11_filter(lev)
    assert engle_ng_test(z) == engle_ng_test(z)


def test_bench_schema_and_score() -> None:
    r = bench_engle_ng()
    for k in (
        "synthetic_lm_lev",
        "synthetic_p_lev",
        "synthetic_p_sym",
        "synthetic_b_sign",
        "synthetic_r2_lev",
        "synthetic_score",
    ):
        assert np.isfinite(r[k])
    assert r["synthetic_score"] == 1.0
