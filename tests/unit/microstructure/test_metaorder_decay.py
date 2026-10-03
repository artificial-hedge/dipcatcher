"""Tests for microstructure/metaorder_decay.py — transient impact split."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.microstructure.metaorder_decay import (
    METAORDER_DECAY_SCHEMA,
    fit_reversion_exponent,
    metaorder_decay_bench,
    run_metaorder,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig


def _cfg(seed: int = 5000, **kw: object) -> ZILobConfig:
    return ZILobConfig(seed=seed, init_depth=8, band=8, **kw)  # type: ignore[arg-type]


def test_metaorder_deterministic() -> None:
    a = run_metaorder(config=_cfg(), side="buy", qty=4, pace_steps=3, warmup=100)
    b = run_metaorder(config=_cfg(), side="buy", qty=4, pace_steps=3, warmup=100)
    assert a.peak == b.peak and a.perm_share == b.perm_share
    np.testing.assert_array_equal(a.curve, b.curve)


def test_frozen_ref_reverts_more_than_touch() -> None:
    t = run_metaorder(config=_cfg(anchor="touch"), side="buy", qty=6, pace_steps=3)
    f = run_metaorder(config=_cfg(anchor="ref"), side="buy", qty=6, pace_steps=3)
    assert f.perm_share < t.perm_share


def test_buy_impact_is_positive_peak() -> None:
    r = run_metaorder(config=_cfg(), side="buy", qty=6, pace_steps=3)
    assert r.peak > 0.0


def test_sell_impact_sign_normalized() -> None:
    r = run_metaorder(config=_cfg(), side="sell", qty=6, pace_steps=3)
    assert r.peak > 0.0  # sign-normalized: a sell's peak is the downward walk


def test_run_metaorder_validates() -> None:
    with pytest.raises(ValueError, match="qty"):
        run_metaorder(config=_cfg(), side="buy", qty=0, pace_steps=1)
    with pytest.raises(ValueError, match="side"):
        run_metaorder(config=_cfg(), side="hold", qty=1, pace_steps=1)


def test_fit_reversion_exponent_recovers_powerlaw() -> None:
    lags = np.asarray([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.float64)
    curve = lags ** (-0.4)
    gamma = fit_reversion_exponent(curve, lags)
    assert gamma == pytest.approx(0.4, abs=0.02)


def test_fit_reversion_exponent_sparse_curve_nan() -> None:
    lags = np.asarray([1, 2, 4, 8], dtype=np.float64)
    curve = np.asarray([0.0, 0.0, 0.0, 0.0])
    assert math.isnan(fit_reversion_exponent(curve, lags))


def test_bench_smoke_and_schema() -> None:
    out = metaorder_decay_bench(n_trials=2, qty=4, pace_steps=3, warmup=60)
    assert out["schema"] == METAORDER_DECAY_SCHEMA
    for arm in ("anchor_touch", "anchor_ref_frozen", "anchor_ref_track"):
        assert arm in out["arms"]
        assert len(out["arms"][arm]["curve_mean"]) == 8
    assert out["data_label"] == "SYNTHETIC"
    assert isinstance(out["payload_sha256"], str) and len(out["payload_sha256"]) == 64
