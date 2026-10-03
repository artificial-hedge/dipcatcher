"""Tests for streak_calibrate — purity knob + calibrated split arm."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig, ZILobSimulator


class TestPurity:
    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="purity"):
            SplitFlow(purity=0.4)
        with pytest.raises(ValueError, match="purity"):
            SplitFlow(purity=1.5)
        with pytest.raises(ValueError, match="purity"):
            SplitFlow(purity=float("nan"))

    def test_default_is_bit_identical(self) -> None:
        a = SplitFlow(p_start=0.5, k_min=5, k_max=20, seed=3)
        b = SplitFlow(p_start=0.5, k_min=5, k_max=20, purity=1.0, seed=3)
        for _ in range(300):
            assert a.current().p_buy == b.current().p_buy
            a.advance()
            b.advance()
        assert a.children_sizes == b.children_sizes

    def test_impure_parent_emits_counterflow(self) -> None:
        f = SplitFlow(p_start=1.0, k_min=50, k_max=50, purity=0.7, seed=1)
        f.advance()  # start a parent (p_start=1 fires on the first call)
        for _ in range(49):
            st = f.current()
            assert st.name.startswith("parent_")
            # Impure: p_buy is 0.7 or 0.3 — never a hard 0/1 stream.
            assert min(abs(st.p_buy - 0.7), abs(st.p_buy - 0.3)) < 1e-12
            f.advance()


class TestCalibratedArm:
    def test_long_runs_emerge(self) -> None:
        sim = ZILobSimulator(
            ZILobConfig(seed=9),
            flow=SplitFlow(
                p_start=0.07,
                size_tail=0.9,
                k_min=70,
                k_max=1800,
                intensity_mult=1.9,
                purity=0.96,
                seed=9,
            ),
        )
        for _ in range(20000):
            sim.step()
        signs = np.asarray([1 if t.aggressor == "buy" else -1 for t in sim.trades])
        # lag-1 autocorr remains in the real-tape band
        x = signs[:-1] - signs[:-1].mean()
        y = signs[1:] - signs[1:].mean()
        lag1 = float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
        assert 0.5 < lag1 < 0.9


def test_bench_smoke() -> None:
    from quant_fund.microstructure.streak_calibrate_bench import (
        streak_calibrate_bench,
    )

    p = streak_calibrate_bench(seed=9)
    assert p["schema"] == "streak_calibrate.v1"
    assert p["data_label"] == "MIXED"
    assert p["research_only"] is True
    assert all(p["claims"].values())
