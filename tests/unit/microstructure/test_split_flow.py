"""Tests for microstructure/split_flow.py + split_flow_bench.py."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.split_flow import SplitFlow, sign_autocorr_curve
from quant_fund.microstructure.split_flow_bench import (
    SPLIT_FLOW_SCHEMA,
    split_flow_bench,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig, ZILobSimulator


def test_split_flow_fail_closed() -> None:
    with pytest.raises(ValueError, match="p_start"):
        SplitFlow(p_start=1.5)
    with pytest.raises(ValueError, match="size_tail"):
        SplitFlow(size_tail=0.0)
    with pytest.raises(ValueError, match="size clip"):
        SplitFlow(k_min=5, k_max=2)
    with pytest.raises(ValueError, match="intensity_mult"):
        SplitFlow(intensity_mult=0.0)


def test_parent_runs_same_sign() -> None:
    flow = SplitFlow(p_start=1.0, k_min=5, k_max=5, seed=1)
    flow.advance()  # starts a parent
    st = flow.current()
    assert st.p_buy in (0.0, 1.0)
    for _ in range(4):
        flow.advance()
        assert flow.current().p_buy == st.p_buy
    flow.advance()  # 5th child consumed
    assert flow.current().p_buy == 0.5


def test_split_flow_drives_sim() -> None:
    sim = ZILobSimulator(ZILobConfig(seed=5), flow=SplitFlow(p_start=0.2, seed=3))
    for _ in range(500):
        sim.step()
    assert sim.n_fills > 0


def test_autocorr_curve() -> None:
    signs = np.array([1.0] * 50 + [-1.0] * 50)
    out = sign_autocorr_curve(signs, (1, 4, 60))
    assert out["lag1"] > 0.9
    assert out["lag4"] > 0.8
    assert np.isnan(out["lag60"]) or out["lag60"] == out["lag60"]


def test_bench_schema_and_persistence() -> None:
    out = split_flow_bench(horizon=4000, seed=7)
    assert out["schema"] == SPLIT_FLOW_SCHEMA
    assert out["data_label"] == "SYNTHETIC"
    arms = {a["name"]: a for a in out["arms"]}
    assert set(arms) == {"iid", "regime", "split"}
    assert arms["split"]["autocorr"]["lag1"] > arms["iid"]["autocorr"]["lag1"] + 0.1
    assert len(out["receipt_sha256"]) == 64
