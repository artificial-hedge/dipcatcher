"""Wave-918 data-structures-4/geometry-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.fractional_cascade import bench_fractional_cascade
from quant_fund.models.free_list import bench_free_list
from quant_fund.models.halfplane_isect import bench_halfplane_isect
from quant_fund.models.object_pool import bench_object_pool
from quant_fund.models.range_min_query import bench_range_min_query
from quant_fund.models.welzl_circle import bench_welzl_circle

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


def bench_fractional_cascade_family(seed: int = _SEED + 30600) -> dict[str, float]:
    return _finite_blob(bench_fractional_cascade(seed))


def bench_range_min_query_family(seed: int = _SEED + 30601) -> dict[str, float]:
    return _finite_blob(bench_range_min_query(seed))


def bench_free_list_family(seed: int = _SEED + 30602) -> dict[str, float]:
    return _finite_blob(bench_free_list(seed))


def bench_object_pool_family(seed: int = _SEED + 30603) -> dict[str, float]:
    return _finite_blob(bench_object_pool(seed))


def bench_welzl_circle_family(seed: int = _SEED + 30604) -> dict[str, float]:
    return _finite_blob(bench_welzl_circle(seed))


def bench_halfplane_isect_family(seed: int = _SEED + 30605) -> dict[str, float]:
    return _finite_blob(bench_halfplane_isect(seed))
