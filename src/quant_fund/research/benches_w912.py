"""Wave-912 spatial-index-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.hilbert_curve import bench_hilbert_curve
from quant_fund.models.morton_order import bench_morton_order
from quant_fund.models.octree_index import bench_octree_index
from quant_fund.models.range_tree import bench_range_tree
from quant_fund.models.rstar_tree import bench_rstar_tree
from quant_fund.models.z_curve import bench_z_curve

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


def bench_octree_index_family(seed: int = _SEED + 30000) -> dict[str, float]:
    return _finite_blob(bench_octree_index(seed))


def bench_range_tree_family(seed: int = _SEED + 30001) -> dict[str, float]:
    return _finite_blob(bench_range_tree(seed))


def bench_hilbert_curve_family(seed: int = _SEED + 30002) -> dict[str, float]:
    return _finite_blob(bench_hilbert_curve(seed))


def bench_z_curve_family(seed: int = _SEED + 30003) -> dict[str, float]:
    return _finite_blob(bench_z_curve(seed))


def bench_morton_order_family(seed: int = _SEED + 30004) -> dict[str, float]:
    return _finite_blob(bench_morton_order(seed))


def bench_rstar_tree_family(seed: int = _SEED + 30005) -> dict[str, float]:
    return _finite_blob(bench_rstar_tree(seed))
