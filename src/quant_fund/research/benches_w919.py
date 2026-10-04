"""Wave-919 computational-geometry-4 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.convex_layers import bench_convex_layers
from quant_fund.models.delaunay_flip import bench_delaunay_flip
from quant_fund.models.polygon_offset import bench_polygon_offset
from quant_fund.models.rotating_sweep import bench_rotating_sweep
from quant_fund.models.visibility_graph import bench_visibility_graph
from quant_fund.models.voronoi_lite import bench_voronoi_lite

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in blob.items():
        if key.lower() in _FORBIDDEN:
            raise ValueError(f"forbidden metric key: {key}")
        if not math.isfinite(val):
            raise ValueError(f"non-finite metric: {key}")
        out[key] = float(val)
    return out


def _floats(xs: Iterable[float]) -> list[float]:
    return [float(x) for x in xs]


def bench_voronoi_lite_family(seed: int = _SEED + 30700) -> dict[str, float]:
    return _finite_blob(bench_voronoi_lite(seed))


def bench_delaunay_flip_family(seed: int = _SEED + 30701) -> dict[str, float]:
    return _finite_blob(bench_delaunay_flip(seed))


def bench_convex_layers_family(seed: int = _SEED + 30702) -> dict[str, float]:
    return _finite_blob(bench_convex_layers(seed))


def bench_polygon_offset_family(seed: int = _SEED + 30703) -> dict[str, float]:
    return _finite_blob(bench_polygon_offset(seed))


def bench_rotating_sweep_family(seed: int = _SEED + 30704) -> dict[str, float]:
    return _finite_blob(bench_rotating_sweep(seed))


def bench_visibility_graph_family(seed: int = _SEED + 30705) -> dict[str, float]:
    return _finite_blob(bench_visibility_graph(seed))
