"""Tests for microstructure/microprice.py — Stoikov imbalance signal."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.microprice import (
    MICROPRICE_SCHEMA,
    collect_imbalance,
    microprice_bench,
    ols_fit,
    touch_imbalance,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig, ZILobSimulator


def test_touch_imbalance_bounds() -> None:
    sim = ZILobSimulator(ZILobConfig(seed=1, init_depth=8, band=8))
    for _ in range(50):
        sim.step()
    i = touch_imbalance(sim)
    assert i is not None
    assert 0.0 <= i <= 1.0


def test_ols_fit_perfect_signal() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    y = 3.0 * x
    f = ols_fit(x, y)
    assert f["beta"] == pytest.approx(3.0)
    assert f["r2"] == pytest.approx(1.0)


def test_ols_fit_zero_signal() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    y = rng.normal(0, 1, 500)
    f = ols_fit(x, y)
    assert abs(f["beta"]) < 0.2
    assert f["r2"] < 0.05


def test_ols_fit_degenerate() -> None:
    f = ols_fit(np.ones(5), np.ones(5))
    assert np.isnan(f["beta"])


def test_collect_imbalance_deterministic() -> None:
    cfg = ZILobConfig(seed=7, init_depth=8, band=8)
    a = collect_imbalance(config=cfg, horizon=200.0)
    b = collect_imbalance(config=cfg, horizon=200.0)
    np.testing.assert_array_equal(a.imb, b.imb)
    np.testing.assert_array_equal(a.dmid, b.dmid)
    np.testing.assert_array_equal(a.signs, b.signs)


def test_collect_imbalance_alignment() -> None:
    tape = collect_imbalance(config=ZILobConfig(seed=9, init_depth=8, band=8), horizon=300.0)
    assert tape.imb.size == tape.dmid.size == tape.signs.size
    assert np.all((tape.imb >= 0.0) & (tape.imb <= 1.0))
    assert set(np.unique(tape.signs)) <= {-1.0, 1.0}


def test_bench_smoke_and_schema() -> None:
    out = microprice_bench(n_seeds=2, horizon=250.0)
    assert out["schema"] == MICROPRICE_SCHEMA
    for arm in ("calm", "trend"):
        assert "r2_imbalance" in out["arms"][arm]
        assert "r2_sign" in out["arms"][arm]
    assert out["data_label"] == "SYNTHETIC"
    assert len(out["payload_sha256"]) == 64
