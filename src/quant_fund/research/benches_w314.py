"""Wave-314 geometry-processing canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.catmull_clark import bench_catmull_clark
from quant_fund.models.half_edge import bench_half_edge
from quant_fund.models.laplacian_smooth import bench_laplacian_smooth
from quant_fund.models.loop_subdiv import bench_loop_subdiv
from quant_fund.models.marching_cubes import bench_marching_cubes
from quant_fund.models.nurbs_eval import bench_nurbs_eval

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


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


def bench_nurbs_eval_family(seed: int = _SEED + 1791) -> dict[str, float]:
    return _floats(_finite_blob("nurbs_eval", bench_nurbs_eval(seed)))


def bench_catmull_clark_family(seed: int = _SEED + 1792) -> dict[str, float]:
    return _floats(_finite_blob("catmull_clark", bench_catmull_clark(seed)))


def bench_loop_subdiv_family(seed: int = _SEED + 1793) -> dict[str, float]:
    return _floats(_finite_blob("loop_subdiv", bench_loop_subdiv(seed)))


def bench_marching_cubes_family(seed: int = _SEED + 1794) -> dict[str, float]:
    return _floats(_finite_blob("marching_cubes", bench_marching_cubes(seed)))


def bench_half_edge_family(seed: int = _SEED + 1795) -> dict[str, float]:
    return _floats(_finite_blob("half_edge", bench_half_edge(seed)))


def bench_laplacian_smooth_family(seed: int = _SEED + 1796) -> dict[str, float]:
    return _floats(_finite_blob("laplacian_smooth", bench_laplacian_smooth(seed)))
