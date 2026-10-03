"""sweep_width_bench contracts."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.sweep_width_bench import (
    SWEEP_WIDTH_BENCH_SCHEMA,
    sweep_width_bench,
)


class TestSweepWidthBench:
    @pytest.mark.slow
    def test_bench_seals_and_claims(self) -> None:
        p = sweep_width_bench(horizon=1500.0, seed=13)
        assert p["schema"] == SWEEP_WIDTH_BENCH_SCHEMA
        assert p["research_only"] is True
        assert p["arms"]["unit_lot"]["p_ge2"] == 0.0
        assert p["claims"]["sized_emerges_multi_level"] is True
        assert p["claims"]["sized_max_realistic"] is True
        assert any("sized_p_ge2" in d for d in p["divergences"])
