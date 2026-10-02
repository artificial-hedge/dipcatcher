"""Tests for Frenet-frame optimal trajectory planning."""

from __future__ import annotations

import numpy as np

from quant_fund.models.frenet import (
    _eval,
    bench_frenet,
    frenet_plan,
    quintic_poly,
)


def test_quintic_boundary_conditions():
    c = quintic_poly(0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 2.0)
    assert abs(_eval(c, 0.0) - 0.1) < 1e-12
    assert abs(_eval(c, 2.0) - 0.3) < 1e-9
    assert abs(_eval(c, 2.0, 1)) < 1e-9
    assert abs(_eval(c, 2.0, 2)) < 1e-9


def test_frenet_picks_centerline():
    path = np.stack([np.linspace(0, 10, 50), np.zeros(50)], axis=1)
    cost, traj, nf = frenet_plan(path, s0=0.0, d0=0.0, v0=0.5)
    assert nf > 0
    assert len(traj) > 0
    # zero offset is the cheapest lateral end → tiny end offset
    assert abs(traj[-1, 1]) <= 0.16


def test_frenet_dodges_obstacle():
    path = np.stack([np.linspace(0, 10, 50), np.zeros(50)], axis=1)
    cost, traj, nf = frenet_plan(
        path,
        s0=0.0,
        d0=0.0,
        v0=0.5,
        obstacles=[(1.8, 0.0), (2.6, 0.05)],
    )
    assert len(traj) > 0
    clr = min(
        np.min(np.hypot(traj[:, 0] - os_, traj[:, 1] - od)) for os_, od in [(1.8, 0.0), (2.6, 0.05)]
    )
    assert clr >= 0.1


def test_bench_frenet():
    out = bench_frenet(seed=20261231)
    assert out["synthetic_frenet_feasible"] > 0
    assert out["synthetic_frenet_obs_clearance"] >= 0.1
    assert all(np.isfinite(v) for v in out.values())
