"""Wave-261 robotics-2 canon adapter: SYNTHETIC benches."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ekf_slam import bench_ekf_slam
from quant_fund.models.frontier_explore import bench_frontier_explore
from quant_fund.models.occupancy_grid import bench_occupancy_grid
from quant_fund.models.particle_slam import bench_particle_slam
from quant_fund.models.pure_pursuit import bench_pure_pursuit
from quant_fund.models.stanley import bench_stanley

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_ekf_slam_family(seed: int = _SEED + 1380) -> dict[str, float]:
    return bench_ekf_slam(seed)


def bench_occupancy_grid_family(seed: int = _SEED + 1381) -> dict[str, float]:
    return bench_occupancy_grid(seed)


def bench_pure_pursuit_family(seed: int = _SEED + 1382) -> dict[str, float]:
    return bench_pure_pursuit(seed)


def bench_stanley_family(seed: int = _SEED + 1383) -> dict[str, float]:
    return bench_stanley(seed)


def bench_particle_slam_family(seed: int = _SEED + 1384) -> dict[str, float]:
    return bench_particle_slam(seed)


def bench_frontier_explore_family(seed: int = _SEED + 1385) -> dict[str, float]:
    return bench_frontier_explore(seed)
