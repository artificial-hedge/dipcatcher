"""release_chase bench — chased orders churn out instead of pinning the touch."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

import quant_fund.microstructure.release_chase_bench as m
from quant_fund.microstructure.release_chase_bench import (
    RELEASE_CHASE_SCHEMA,
    release_chase_bench,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator, santa_fe_config
from quant_fund.research.receipt_v2 import verify_receipt_file


def _drive(sim: ZILobSimulator, n: int) -> None:
    for _ in range(n):
        sim.step()


def _fill_then_drive(sim: ZILobSimulator, n: int) -> None:
    """Trigger one post-fill marker, then run ``n`` events."""
    sim.inject_market_order("buy", qty=1)
    _drive(sim, n)


def test_chase_release_bit_identical_at_zero() -> None:
    """release=0.9 with no chase stock never draws — bit-identical."""
    c0 = santa_fe_config(seed=11)
    c1 = dataclasses.replace(santa_fe_config(seed=11), chase_release=0.9)
    s0, s1 = ZILobSimulator(c0), ZILobSimulator(c1)
    _drive(s0, 3000)
    _drive(s1, 3000)
    assert s0.n_events == s1.n_events
    assert s0.n_fills == s1.n_fills
    assert s0.n_lo_arrivals == s1.n_lo_arrivals


def test_chase_orders_are_tagged_and_tracked() -> None:
    cfg = dataclasses.replace(santa_fe_config(seed=11), unhit_imp_frac=1.0, unhit_imp_window=400)
    sim = ZILobSimulator(cfg)
    _fill_then_drive(sim, 2000)
    assert sim._chase_oids or sim.n_cancellations > 0
    for oid in sim._chase_oids:
        assert sim._orders[oid].tag == "chase"


def test_release_consumes_chase_orders_faster() -> None:
    """release=1.0 must cancel chase orders as fast as cancels arrive."""

    def chase_stock(release: float) -> int:
        cfg = dataclasses.replace(
            santa_fe_config(seed=11),
            unhit_imp_frac=1.0,
            unhit_imp_window=2000,
            chase_release=release,
        )
        sim = ZILobSimulator(cfg)
        _fill_then_drive(sim, 4000)
        return len(sim._chase_oids)

    no_rel = chase_stock(0.0)
    rel = chase_stock(1.0)
    assert no_rel > 0
    assert rel < no_rel


def test_release_picks_chase_victim() -> None:
    """At release=1.0 with chase stock present, a cancel event frees a chase order."""
    cfg = dataclasses.replace(
        santa_fe_config(seed=11),
        unhit_imp_frac=1.0,
        unhit_imp_window=2000,
        chase_release=1.0,
    )
    sim = ZILobSimulator(cfg)
    _fill_then_drive(sim, 3000)
    # Every surviving member of the set is a chase-tagged live order.
    assert all(sim._orders[o].tag == "chase" for o in sim._chase_oids)


def test_chase_release_validates_probability() -> None:
    with pytest.raises(ValueError, match="chase_release"):
        dataclasses.replace(santa_fe_config(seed=11), chase_release=1.5)


def test_release_bench_seals_and_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seal a fabricated grid: one cell must land all five targets."""

    def fake_cell(
        cooldown: int,
        frac: float,
        imp_window: int,
        release: float,
        cxl_damp: float,
        cxl_window: int,
        *,
        horizon: int,
        seed: int,
    ) -> dict[str, Any]:
        hit = release > 0.0 and cooldown > 0 and frac > 0.0
        return {
            "refill_cooldown": cooldown,
            "unhit_imp_frac": frac,
            "unhit_imp_window": imp_window,
            "chase_release": release,
            "cxl_unhit_damp": cxl_damp,
            "cxl_unhit_window": cxl_window,
            "instant_signed_ticks": m._INSTANT_TAPE if hit else 0.0,
            "k200_ticks": m._K200_TAPE if hit else 0.0,
            "lo_channel_ticks": m._LO_CH_TAPE if hit else 0.0,
            "fill_channel_ticks": m._FILL_CH_TAPE if hit else 0.0,
            "cxl_channel_ticks": m._CXL_CH_TAPE if hit else 0.0,
        }

    monkeypatch.setattr(m, "_release_cell", fake_cell)
    payload = release_chase_bench(horizon=500, seed=3)
    assert payload["schema"] == RELEASE_CHASE_SCHEMA
    assert payload["claims"]["joint_closure_exists"] is True
    out = tmp_path / "release_chase_test.json"
    out.write_text(json.dumps(payload))
    assert verify_receipt_file(out)["valid"]


def test_release_bench_zero_cell() -> None:
    payload = release_chase_bench(horizon=1500, seed=5)
    zero = next(
        c
        for c in payload["cells"]
        if c["unhit_imp_frac"] == 0.0 and c["refill_cooldown"] == 0 and c["chase_release"] == 0.0
    )
    assert zero["instant_signed_ticks"] is not None
    assert zero["k200_ticks"] is not None
    assert payload["claims"]["grid_evaluated"]
    assert set(payload["claims"]) == {
        "grid_evaluated",
        "release_preserves_lo",
        "release_recovers_instant",
        "cooldown_lifts_instant",
        "joint_closure_exists",
    }
