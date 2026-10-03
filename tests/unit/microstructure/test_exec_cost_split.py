"""Tests for microstructure/exec_cost_split.py."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.exec_cost_split import (
    EXEC_SPLIT_SCHEMA,
    exec_episode,
    exec_split_bench,
    twap_schedule,
)
from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig


def test_twap_schedule_sums() -> None:
    s = twap_schedule(10, 3)
    assert sum(s) == 10
    assert len(s) == 3
    assert s == [4, 3, 3]
    with pytest.raises(ValueError):
        twap_schedule(0, 3)


def test_episode_fail_closed() -> None:
    with pytest.raises(ValueError, match="side"):
        exec_episode(
            ZILobConfig(seed=1),
            None,
            parent_size=4,
            n_children=2,
            events_per_child=5,
            side="north",  # type: ignore[arg-type]
            episode_seed=0,
        )


def test_episode_conservation() -> None:
    out = exec_episode(
        ZILobConfig(seed=3),
        None,
        parent_size=8,
        n_children=4,
        events_per_child=10,
        side="buy",
        episode_seed=0,
    )
    assert out is not None
    assert 0.0 < out["fill_fraction"] <= 1.0
    assert out["shortfall_ticks"] >= 0.0  # buying pays ask >= mid


def test_bench_schema_and_order() -> None:
    out = exec_split_bench(parent_size=12, n_children=4, events_per_child=15, n_episodes=4, seed=2)
    assert out["schema"] == EXEC_SPLIT_SCHEMA
    assert out["data_label"] == "SYNTHETIC"
    assert set(out["arms"]) == {"iid", "regime", "split"}
    for v in out["arms"].values():
        assert v["n_episodes"] >= 1
        assert v["shortfall_ticks_mean"] >= 0
    assert len(out["receipt_sha256"]) == 64


def test_bench_deterministic() -> None:
    a = exec_split_bench(parent_size=8, n_children=2, events_per_child=10, n_episodes=3, seed=5)
    b = exec_split_bench(parent_size=8, n_children=2, events_per_child=10, n_episodes=3, seed=5)
    assert a["arms"] == b["arms"]


def test_split_flow_accepted_as_flow() -> None:
    # the MOFlow protocol must accept SplitFlow at runtime here
    sim_ok = exec_episode(
        ZILobConfig(seed=9),
        SplitFlow(p_start=0.3, seed=1),
        parent_size=6,
        n_children=3,
        events_per_child=8,
        side="sell",
        episode_seed=0,
    )
    assert sim_ok is None or sim_ok["shortfall_ticks"] >= 0
