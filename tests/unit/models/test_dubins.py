"""Tests for Dubins shortest paths."""

from __future__ import annotations

import math

import numpy as np

from quant_fund.models.dubins import bench_dubins, dubins_path, dubins_sample


def test_dubins_straight():
    L, _ = dubins_path((0, 0, 0), (4, 0, 0), rho=1.0)
    assert abs(L - 4.0) < 1e-9


def test_dubins_reaches_goal():
    L, segs = dubins_path((0, 0, 0.3), (3, 1, 0.1), rho=1.5)
    pts = dubins_sample((0, 0, 0.3), segs, rho=1.5, step=0.01)
    assert np.abs(pts[-1, :2] - np.array([3.0, 1.0])).max() < 0.05
    assert abs(((pts[-1, 2] - 0.1 + math.pi) % (2 * math.pi)) - math.pi) < 0.05


def test_dubins_turnaround():
    L, segs = dubins_path((0, 0, 0), (0.5, 0, math.pi), rho=1.0)
    pts = dubins_sample((0, 0, 0), segs, rho=1.0, step=0.01)
    # must end heading backwards
    h = pts[-1, 2] % (2 * math.pi)
    assert abs(h - math.pi) < 0.1
    assert L > 2.0  # can't be a straight line


def test_dubins_symmetry():
    L1, _ = dubins_path((0, 0, 0.3), (3, 1, 0.1), rho=1.5)
    L2, _ = dubins_path((3, 1, 0.1 + math.pi), (0, 0, 0.3 + math.pi), rho=1.5)
    assert abs(L1 - L2) < 1e-9


def test_bench_dubins_keys():
    out = bench_dubins(seed=20261231)
    assert out["synthetic_dubins_straight_err"] < 1e-9
    assert out["synthetic_dubins_endpos_err"] < 0.05
    assert all(np.isfinite(v) for v in out.values())
