"""Tests for reseed_hazard_bench."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.reseed_hazard_bench import (
    RESEED_HAZARD_SCHEMA,
    reseed_hazard_bench,
)


@pytest.mark.slow
def test_bench_sim_only() -> None:
    out = reseed_hazard_bench(tape_dir=None, horizon=8000)
    assert out["schema"] == RESEED_HAZARD_SCHEMA
    assert out["research_only"] is True
    assert out["tape"] is None
    assert len(out["sim_arms"]) == 5
    for arm in out["sim_arms"]:
        assert arm["n_emptied"] > 0
        if arm["reseed_rate_500"] is not None:
            assert 0.0 <= arm["reseed_rate_500"] <= 1.0


def test_stats_shape() -> None:
    from quant_fund.microstructure.reseed_hazard_bench import _stats

    s = _stats(10, [5, 10, 20], 2)
    assert s["reseed_rate_500"] == 0.3
    assert s["reseed_latency_p50"] == 10.0
    assert s["reseed_as_touch_share"] == round(2 / 3, 4)
    assert _stats(0, [], 0)["reseed_rate_500"] is None
