"""initiative_fade_bench + p_buy_drift contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from quant_fund.microstructure.initiative_fade_bench import (
    INITIATIVE_FADE_SCHEMA,
    initiative_fade_bench,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator, santa_fe_config


def _trades(sim: ZILobSimulator) -> list[tuple[float, str, int]]:
    return [(tr.t, tr.aggressor, tr.level) for tr in sim.trades]


class TestPBuyDrift:
    def test_zero_drift_bit_identical(self) -> None:
        a = ZILobSimulator(santa_fe_config(seed=11))
        b = ZILobSimulator(replace(santa_fe_config(seed=11), p_buy_drift=0.0))
        for _ in range(500):
            a.step()
            b.step()
        assert _trades(a) == _trades(b)
        assert a.n_events == b.n_events
        assert a.t == b.t

    def test_positive_drift_raises_late_buy_share(self) -> None:
        flat = ZILobSimulator(santa_fe_config(seed=5, p_buy=0.5))
        drift = ZILobSimulator(replace(santa_fe_config(seed=5, p_buy=0.5), p_buy_drift=2e-4))
        while flat.t < 800.0:
            flat.step()
        while drift.t < 800.0:
            drift.step()

        def late_share(sim: ZILobSimulator) -> float:
            fills = [tr for tr in sim.trades if tr.t >= 400.0]
            return sum(1 for tr in fills if tr.aggressor == "buy") / len(fills)

        assert late_share(drift) > late_share(flat)

    def test_clip_bounds_saturate_side_draw(self) -> None:
        down = ZILobSimulator(replace(santa_fe_config(seed=7, p_buy=0.5), p_buy_drift=-1.0))
        up = ZILobSimulator(replace(santa_fe_config(seed=7, p_buy=0.5), p_buy_drift=1.0))
        while down.t < 50.0:
            down.step()
        while up.t < 50.0:
            up.step()
        assert all(tr.aggressor == "sell" for tr in down.trades if tr.t > 1.0)
        assert all(tr.aggressor == "buy" for tr in up.trades if tr.t > 1.0)


class TestBench:
    @pytest.mark.slow
    def test_bench_seals(self) -> None:
        p = initiative_fade_bench(horizon=1500.0, seed=13)
        assert p["schema"] == INITIATIVE_FADE_SCHEMA
        assert p["research_only"] is True
        assert p["data_label"] == "MIXED"
        assert len(p["arms"]["flat"]["thirds"]) == 3
        assert p["claims"]["flat_is_flat"] is True
        assert p["claims"]["drift_fades"] is True
        assert p["claims"]["drift_early_on_tape"] is True
        assert p["claims"]["drift_late_on_tape"] is True
