"""Unit tests for quant_fund.models.obizhaeva_wang."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.obizhaeva_wang import (
    bench_obizhaeva_wang,
    ow_cost,
    ow_optimal_cost,
    ow_schedule,
)


def test_schedule_conserves_mass() -> None:
    block, rate, _ = ow_schedule(1.0, 2.0, 3.0)
    # 2 blocks + rate*T == X
    assert 2 * block + rate * 2.0 == pytest.approx(1.0)


def test_block_rate_identity() -> None:
    block, rate, _ = ow_schedule(1.0, 1.0, 2.0)
    assert rate == pytest.approx(2.0 * block)


def test_cost_decreases_in_resilience() -> None:
    c1 = ow_optimal_cost(1.0, 1.0, 0.1, 1.0)
    c4 = ow_optimal_cost(1.0, 1.0, 0.1, 4.0)
    assert c4 < c1


def test_twap_baseline() -> None:
    x = np.full(50, 0.02)
    c = ow_cost(x, 0.02, 0.0, 2.0)
    assert np.isfinite(c) and c > 0


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        ow_schedule(0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        ow_cost(np.array([0.1]), -1.0, 0.1, 1.0)


def test_bench_score() -> None:
    out = bench_obizhaeva_wang()
    assert out["score"] == 1.0
    assert out["synthetic_ow_twap_gain"] > 0.0
