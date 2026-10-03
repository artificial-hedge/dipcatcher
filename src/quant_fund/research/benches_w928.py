"""Wave-928 computational-geometry-6 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.beta_skeleton import bench_beta_skeleton
from quant_fund.models.convex_hull_3d import bench_convex_hull_3d
from quant_fund.models.medial_axis import bench_medial_axis
from quant_fund.models.polygon_boolean import bench_polygon_boolean
from quant_fund.models.polygon_centroid import bench_polygon_centroid
from quant_fund.models.shape_context import bench_shape_context

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


def bench_convex_hull_3d_family(seed: int = _SEED + 31600) -> dict[str, float]:
    return _finite_blob(bench_convex_hull_3d(seed))


def bench_polygon_boolean_family(seed: int = _SEED + 31601) -> dict[str, float]:
    return _finite_blob(bench_polygon_boolean(seed))


def bench_medial_axis_family(seed: int = _SEED + 31602) -> dict[str, float]:
    return _finite_blob(bench_medial_axis(seed))


def bench_polygon_centroid_family(seed: int = _SEED + 31603) -> dict[str, float]:
    return _finite_blob(bench_polygon_centroid(seed))


def bench_shape_context_family(seed: int = _SEED + 31604) -> dict[str, float]:
    return _finite_blob(bench_shape_context(seed))


def bench_beta_skeleton_family(seed: int = _SEED + 31605) -> dict[str, float]:
    return _finite_blob(bench_beta_skeleton(seed))
