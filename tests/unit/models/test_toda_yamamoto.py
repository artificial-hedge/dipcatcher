"""Tests for toda_yamamoto — augmented-lag Granger MWALD."""

import numpy as np
import pytest

from quant_fund.models.toda_yamamoto import (
    bench_toda_yamamoto,
    granger_mwald,
    synth_ty,
    var_select_order,
)


def test_true_direction_rejected() -> None:
    y = synth_ty(seed=1)
    r = granger_mwald(y, cause=0, target=1, p=1, d=1)
    assert r["df"] == 1.0
    assert r["p_value"] < 0.01
    assert r["wald"] > 10.0


def test_false_direction_retained() -> None:
    y = synth_ty(seed=1)
    r = granger_mwald(y, cause=1, target=0, p=1, d=1)
    assert r["p_value"] > 0.01


def test_no_link_both_retained() -> None:
    y = synth_ty(seed=4, g=0.0)
    r = granger_mwald(y, cause=0, target=1, p=1, d=1)
    assert r["p_value"] > 0.01


def test_var_order_selection() -> None:
    y = synth_ty(seed=2)
    p = var_select_order(y, pmax=6)
    assert 1 <= p <= 6


def test_larger_p_still_works() -> None:
    y = synth_ty(seed=6)
    r = granger_mwald(y, cause=0, target=1, p=2, d=1)
    assert r["df"] == 2.0
    assert r["p_value"] < 0.05


def test_fail_closed_inputs() -> None:
    y = synth_ty(seed=1)
    with pytest.raises(ValueError):
        granger_mwald(y, cause=0, target=0, p=1, d=1)
    with pytest.raises(ValueError):
        granger_mwald(y, cause=5, target=1, p=1, d=1)
    with pytest.raises(ValueError):
        granger_mwald(y, cause=0, target=1, p=0, d=1)
    with pytest.raises(ValueError):
        granger_mwald(np.ones((50, 1)), cause=0, target=0, p=1, d=1)
    bad = y.copy()
    bad[10, 0] = np.nan
    with pytest.raises(ValueError):
        granger_mwald(bad, cause=0, target=1, p=1, d=1)
    with pytest.raises(ValueError):
        granger_mwald(y[:14], cause=0, target=1, p=2, d=1)


def test_determinism() -> None:
    y = synth_ty(seed=8)
    a = granger_mwald(y, cause=0, target=1, p=1, d=1)
    b = granger_mwald(y, cause=0, target=1, p=1, d=1)
    assert a["wald"] == b["wald"]
    assert a["p_value"] == b["p_value"]


def test_bench_schema_and_score() -> None:
    r = bench_toda_yamamoto(seed=3)
    for k in ("p_lag", "wald_fwd", "pval_fwd", "wald_rev", "pval_rev", "score"):
        assert np.isfinite(r[k])
    assert r["score"] == 1.0
