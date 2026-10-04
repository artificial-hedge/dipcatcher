"""Wave-920 computational-geometry-5 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.alpha_shape import bench_alpha_shape
from quant_fund.models.diameter_pair import bench_diameter_pair
from quant_fund.models.min_area_rect import bench_min_area_rect
from quant_fund.models.minkowski_sum_poly import bench_minkowski_sum_poly
from quant_fund.models.monotone_partition import bench_monotone_partition
from quant_fund.models.polygon_triangulate import bench_polygon_triangulate

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


def bench_monotone_partition_family(seed: int = _SEED + 30800) -> dict[str, float]:
    return _finite_blob(bench_monotone_partition(seed))


def bench_polygon_triangulate_family(seed: int = _SEED + 30801) -> dict[str, float]:
    return _finite_blob(bench_polygon_triangulate(seed))


def bench_min_area_rect_family(seed: int = _SEED + 30802) -> dict[str, float]:
    return _finite_blob(bench_min_area_rect(seed))


def bench_diameter_pair_family(seed: int = _SEED + 30803) -> dict[str, float]:
    return _finite_blob(bench_diameter_pair(seed))


def bench_alpha_shape_family(seed: int = _SEED + 30804) -> dict[str, float]:
    return _finite_blob(bench_alpha_shape(seed))


def bench_minkowski_sum_poly_family(seed: int = _SEED + 30805) -> dict[str, float]:
    return _finite_blob(bench_minkowski_sum_poly(seed))
