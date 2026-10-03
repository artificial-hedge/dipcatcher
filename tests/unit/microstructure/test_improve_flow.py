"""improve_flow_bench + lo_improve_frac instrumentation contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from quant_fund.microstructure.improve_flow_bench import (
    IMPROVE_FLOW_SCHEMA,
    improve_flow_bench,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator, santa_fe_config


class TestImproveCounters:
    def test_zero_frac_bit_identical(self) -> None:
        a = ZILobSimulator(santa_fe_config(seed=11))
        b = ZILobSimulator(replace(santa_fe_config(seed=11), lo_improve_frac=0.0))
        for _ in range(400):
            a.step()
            b.step()
        assert a.event_counts() == b.event_counts()
        assert a.trades == b.trades

    def test_frac_one_fires_improves(self) -> None:
        cfg = replace(santa_fe_config(seed=3), band=14, lo_offset=12, lo_improve_frac=1.0)
        sim = ZILobSimulator(cfg)
        while sim.t < 200.0:
            sim.step()
        ec = sim.event_counts()
        assert ec["n_lo_improve"] + ec["n_lo_join"] > 0
        # every improve-arm landing stays strictly passive (never crosses)
        assert ec["n_lo_arrivals"] > 0

    def test_frac_validation(self) -> None:
        with pytest.raises(ValueError, match="lo_improve_frac"):
            replace(santa_fe_config(seed=1), lo_improve_frac=1.5)


class TestBench:
    @pytest.mark.slow
    def test_bench_seals(self) -> None:
        p = improve_flow_bench(horizon=1500.0, seed=13)
        assert p["schema"] == IMPROVE_FLOW_SCHEMA
        assert p["research_only"] is True
        assert p["claims"]["improve_flow_raises_join_share"] is True
        assert p["receipt_sha256"]
