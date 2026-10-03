"""Tests for mid_dark — hidden midpoint-pegged depth on the ZI-LOB."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from quant_fund.microstructure.mid_dark_bench import (
    _GRID,
    _dark_cell,
    mid_dark_bench,
)
from quant_fund.microstructure.zi_lob_simulator import (
    ZILobSimulator,
    santa_fe_config,
)


def _run(cfg: Any, horizon: float = 400.0) -> ZILobSimulator:
    sim = ZILobSimulator(cfg)
    sim.run(horizon)
    return sim


def test_knob_off_is_bit_identical() -> None:
    a = _run(santa_fe_config(seed=3))
    b = _run(santa_fe_config(seed=3))
    assert [(t.price, t.aggressor) for t in a.trades] == [(t.price, t.aggressor) for t in b.trades]
    assert a.event_counts()["n_dark_fills"] == 0


def test_dark_pegs_fill_at_mid_and_never_enter_book() -> None:
    sim = _run(dataclasses.replace(santa_fe_config(seed=11), mid_dark_frac=0.8), 800.0)
    counts = sim.event_counts()
    assert counts["n_dark_placed"] > 0
    assert counts["n_dark_fills"] > 0
    dark_trades = [t for t in sim.trades if t.maker_tag == "mid_dark"]
    assert dark_trades
    # Dark fills print at the pegged mid: level is the doubled-lattice
    # index bb+ba, so price == s0 + level * tick / 2.
    cfg = sim.cfg
    for t in dark_trades:
        assert t.price == pytest.approx(cfg.s0 + 0.5 * t.level * cfg.tick)
    # Conservation: dark orders never land in the visible book.
    for side in ("buy", "sell"):
        book = sim._asks if side == "sell" else sim._bids  # noqa: SLF001
        for dq in book.values():
            for oid in dq:
                assert sim._orders[oid].tag != "mid_dark"  # noqa: SLF001


def test_dark_pegs_lapse_when_mid_moves() -> None:
    sim = ZILobSimulator(dataclasses.replace(santa_fe_config(seed=5), mid_dark_frac=0.9))
    for _ in range(4000):
        sim.step()
    # Lapses + fills + still-pegged must equal placements.
    c = sim.event_counts()
    assert c["n_dark_placed"] == c["n_dark_fills"] + c["n_dark_lapses"] + sum(
        len(dq)
        for dq in sim._dark.values()  # noqa: SLF001
    )


def test_mid_dark_frac_validation() -> None:
    with pytest.raises(ValueError, match="mid_dark_frac"):
        dataclasses.replace(santa_fe_config(seed=1), mid_dark_frac=1.5)
    with pytest.raises(ValueError, match="mid_dark_ttl"):
        dataclasses.replace(santa_fe_config(seed=1), mid_dark_ttl=-1)


def test_dark_cell_returns_channel_measures() -> None:
    c = _dark_cell(0, 0.4, 0, 0.5, 200, horizon=3000, seed=7)
    for key in (
        "instant_signed_ticks",
        "k200_ticks",
        "lo_channel_ticks",
        "fill_channel_ticks",
        "cxl_channel_ticks",
    ):
        assert key in c
    assert c["n_dark_fills"] >= 0
    assert c["dark_fill_share"] is not None


def test_bench_fabricated_grid_seals(monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_fund.microstructure.mid_dark_bench as m

    def fake_cell(
        cooldown: int,
        dark_frac: float,
        dark_ttl: int,
        imp_frac: float,
        imp_window: int,
        *,
        horizon: int,
        seed: int,
    ) -> dict[str, Any]:
        in_tol = imp_frac > 0.4 and dark_frac > 0.0
        return {
            "refill_cooldown": cooldown,
            "mid_dark_frac": dark_frac,
            "mid_dark_ttl": dark_ttl,
            "unhit_imp_frac": imp_frac,
            "unhit_imp_window": imp_window,
            "n_fills": 100,
            "n_dark_fills": 10 if dark_frac > 0 else 0,
            "dark_fill_share": 0.1 if dark_frac > 0 else 0.0,
            "instant_signed_ticks": 0.89 if in_tol else 0.3,
            "k200_ticks": 4.6 if in_tol else 2.0,
            "k200_per_channel_ticks": {},
            "lo_channel_ticks": 2.8 if in_tol else 0.0,
            "fill_channel_ticks": 2.5 if in_tol else 0.0,
            "cxl_channel_ticks": -0.7,
            "attr_windows": {},
        }

    monkeypatch.setattr(m, "_dark_cell", fake_cell)
    out = mid_dark_bench(horizon=100, seed=1)
    assert out["claims"]["grid_evaluated"]
    assert out["claims"]["dark_channel_active"]
    assert out["claims"]["joint_closure_exists"]
    assert out["receipt_sha256"]
    # Tolerances: only cells with dark+chase land all five.
    n_tol = sum(1 for c in out["grid"] if c["all_in_tol"])
    n_expect = sum(1 for g in _GRID if g[3] > 0.4 and g[1] > 0.0)
    assert n_tol == n_expect


def test_bench_zero_cell_claims(monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_fund.microstructure.mid_dark_bench as m

    def flat_cell(
        cooldown: int,
        dark_frac: float,
        dark_ttl: int,
        imp_frac: float,
        imp_window: int,
        *,
        horizon: int,
        seed: int,
    ) -> dict[str, Any]:
        return {
            "refill_cooldown": cooldown,
            "mid_dark_frac": dark_frac,
            "mid_dark_ttl": dark_ttl,
            "unhit_imp_frac": imp_frac,
            "unhit_imp_window": imp_window,
            "n_fills": 10,
            "n_dark_fills": 0,
            "dark_fill_share": 0.0,
            "instant_signed_ticks": 0.0,
            "k200_ticks": 0.0,
            "k200_per_channel_ticks": {},
            "lo_channel_ticks": 0.0,
            "fill_channel_ticks": 0.0,
            "cxl_channel_ticks": 0.0,
            "attr_windows": {},
        }

    monkeypatch.setattr(m, "_dark_cell", flat_cell)
    out = mid_dark_bench(horizon=100, seed=1)
    assert not out["claims"]["dark_channel_active"]
    assert not out["claims"]["joint_closure_exists"]
    assert not out["claims"]["dark_preserves_lo"]
