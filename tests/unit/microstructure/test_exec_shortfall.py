"""Tests for microstructure/exec_shortfall.py — AC trajectory execution on ZI-LOB."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.exec_shortfall import (
    EXEC_SHORTFALL_SCHEMA,
    ac_trajectory,
    exec_shortfall_bench,
    run_execution,
    twap_trajectory,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig, ZILobSimulator


def test_ac_trajectory_sums_and_front_loads() -> None:
    t = ac_trajectory(60, 12, kappa=0.35)
    assert int(np.sum(t)) == 60
    assert len(t) == 12
    # cumulative rounding preserves the sum; the schedule front-loads
    assert t[0] > t[-1]
    assert t[0] > np.median(t[1:])
    assert np.mean(t[:6]) > np.mean(t[6:])
    # larger kappa -> steeper front-load
    steep = ac_trajectory(60, 12, kappa=1.0)
    assert steep[0] > t[0]


def test_twap_trajectory_uniform_and_exact() -> None:
    t = twap_trajectory(61, 12)
    assert int(np.sum(t)) == 61
    assert t[-1] == 6  # remainder lands on the last slice
    assert np.all(t[:-1] == 5)


def test_run_execution_measures_buy_shortfall() -> None:
    sim = ZILobSimulator(ZILobConfig(seed=0))
    for _ in range(400):
        sim.step()
    sched = twap_trajectory(20, 4)
    res = run_execution(sim, side="buy", schedule=sched, pace=30)
    assert res.is_ticks > 0  # buying walks the ask — positive slippage
    assert res.n_lots_filled == 20
    assert res.avg_price >= res.arrival_mid


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        ac_trajectory(0, 4, kappa=0.5)
    with pytest.raises(ValueError):
        ac_trajectory(4, 8, kappa=0.5)  # fewer lots than slices
    with pytest.raises(ValueError):
        ac_trajectory(10, 4, kappa=0.0)
    with pytest.raises(ValueError):
        twap_trajectory(2, 4)
    sim = ZILobSimulator(ZILobConfig(seed=0))
    with pytest.raises(ValueError):
        run_execution(sim, side="buy", schedule=twap_trajectory(4, 2), pace=0)


def test_bench_schema_determinism_and_frontload_edge() -> None:
    r1 = exec_shortfall_bench(parent_qty=30, n_slices=6, pace=25, n_seeds=4)
    r2 = exec_shortfall_bench(parent_qty=30, n_slices=6, pace=25, n_seeds=4)
    assert r1["payload_sha256"] == r2["payload_sha256"]
    assert r1["schema"] == EXEC_SHORTFALL_SCHEMA
    assert r1["kind"] == "exec_shortfall"
    assert r1["data_label"] == "SYNTHETIC"
    assert r1["research_only"] is True
    assert set(r1["density_cells"]) == {"beta_0", "beta_1"}
    # mechanism: pace gaps let the book replenish, so front-loading buys
    # earlier into a rebuilt book — lower mean IS in both regimes
    for cell in r1["density_cells"].values():
        assert cell["paired_ac_minus_twap"]["mean"] < 0
        assert (
            cell["schedules"]["ac_front"]["is_ticks_q95"]
            < cell["schedules"]["twap"]["is_ticks_q95"]
        )
