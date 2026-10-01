"""maker_age_bench + cxl_ages instrumentation contracts."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.maker_age_bench import MAKER_AGE_SCHEMA, maker_age_bench
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator, santa_fe_config


class TestCxlAges:
    def test_ages_recorded(self) -> None:
        sim = ZILobSimulator(santa_fe_config(seed=3))
        while sim.n_cancellations < 30:
            sim.step()
            assert sim.t < 1e6
        assert len(sim.cxl_ages) == sim.n_cancellations
        assert all(a >= 0.0 for a in sim.cxl_ages)

    def test_bit_identical_log(self) -> None:
        a, b = ZILobSimulator(santa_fe_config(seed=11)), ZILobSimulator(santa_fe_config(seed=11))
        for _ in range(500):
            a.step()
            b.step()
        assert a.cxl_ages == b.cxl_ages


class TestBench:
    @pytest.mark.slow
    def test_bench_seals(self) -> None:
        p = maker_age_bench(horizon=1500.0, seed=13)
        assert p["schema"] == MAKER_AGE_SCHEMA
        assert p["research_only"] is True
        assert p["claims"]["sized_matches_median"] is True
        assert p["claims"]["unit_lot_overshoots_median"] is True
