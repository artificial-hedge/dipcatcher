"""Unit tests for quant_fund.models.transfer_entropy."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.transfer_entropy import (
    bench_transfer_entropy,
    te_both_directions,
    transfer_entropy,
)


def test_direction_recovery() -> None:
    rng = np.random.default_rng(5)
    n = 900
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.standard_normal()
        y[t] = 0.3 * y[t - 1] + 0.9 * x[t - 1] + rng.standard_normal()
    t_xy, t_yx = te_both_directions(x, y, k=4)
    assert t_xy > t_yx
    assert t_xy > 0.05


def test_independent_floor() -> None:
    rng = np.random.default_rng(6)
    x = rng.standard_normal(600)
    y = rng.standard_normal(600)
    assert transfer_entropy(x, y, k=4) < 0.15


def test_input_validation() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(200)
    with pytest.raises(ValueError):
        transfer_entropy(x, rng.standard_normal(100))
    with pytest.raises(ValueError):
        transfer_entropy(x[:20], x[:20])
    with pytest.raises(ValueError):
        transfer_entropy(np.full(200, np.nan), x)


def test_bench_score() -> None:
    out = bench_transfer_entropy()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_te_xy"] > out["synthetic_te_yx"]
