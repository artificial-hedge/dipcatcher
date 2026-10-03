"""Tests for dynamic-window local planning."""

from __future__ import annotations

import numpy as np

from quant_fund.models.dwa import bench_dwa, dwa_drive, naive_drive


def test_dwa_free_space():
    reached, steps, clear = dwa_drive((0.1, 0.1, 0.0), (0.7, 0.7), [])
    assert reached
    assert steps < 300


def test_dwa_avoids_blocker():
    obstacles = [(0.5, 0.45, 0.1), (0.55, 0.65, 0.08)]
    reached, steps, clear = dwa_drive((0.1, 0.1, 0.0), (0.9, 0.9), obstacles)
    assert reached
    assert clear >= 0.0


def test_naive_crashes():
    obstacles = [(0.5, 0.45, 0.1), (0.55, 0.65, 0.08)]
    reached, steps, clear = naive_drive((0.1, 0.1, 0.0), (0.9, 0.9), obstacles)
    assert not reached
    assert clear < 0.0


def test_bench_dwa():
    out = bench_dwa(seed=20261231)
    assert out["synthetic_dwa_reached"] == 1.0
    assert out["synthetic_naive_reached"] == 0.0
    assert out["synthetic_dwa_corridor_reached"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
