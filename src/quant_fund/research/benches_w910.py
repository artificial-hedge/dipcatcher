"""Wave-910 b-tree family canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.b_plus_tree import bench_b_plus_tree
from quant_fund.models.b_star_tree import bench_b_star_tree
from quant_fund.models.b_tree import bench_b_tree
from quant_fund.models.tango_tree import bench_tango_tree
from quant_fund.models.wavl_tree import bench_wavl_tree
from quant_fund.models.weight_balanced_tree import bench_weight_balanced_tree

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


def bench_b_tree_family(seed: int = _SEED + 29800) -> dict[str, float]:
    return _finite_blob(bench_b_tree(seed))


def bench_b_plus_tree_family(seed: int = _SEED + 29801) -> dict[str, float]:
    return _finite_blob(bench_b_plus_tree(seed))


def bench_b_star_tree_family(seed: int = _SEED + 29802) -> dict[str, float]:
    return _finite_blob(bench_b_star_tree(seed))


def bench_weight_balanced_tree_family(seed: int = _SEED + 29803) -> dict[str, float]:
    return _finite_blob(bench_weight_balanced_tree(seed))


def bench_wavl_tree_family(seed: int = _SEED + 29804) -> dict[str, float]:
    return _finite_blob(bench_wavl_tree(seed))


def bench_tango_tree_family(seed: int = _SEED + 29805) -> dict[str, float]:
    return _finite_blob(bench_tango_tree(seed))
