"""Wave-915 BVP/tree-exotics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bvp_eigen import bench_bvp_eigen
from quant_fund.models.continuation_bvp import bench_continuation_bvp
from quant_fund.models.fusion_tree import bench_fusion_tree
from quant_fund.models.loser_tree import bench_loser_tree
from quant_fund.models.robbins_bvp import bench_robbins_bvp
from quant_fund.models.superposition_bvp import bench_superposition_bvp

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


def bench_superposition_bvp_family(seed: int = _SEED + 30300) -> dict[str, float]:
    return _finite_blob(bench_superposition_bvp(seed))


def bench_continuation_bvp_family(seed: int = _SEED + 30301) -> dict[str, float]:
    return _finite_blob(bench_continuation_bvp(seed))


def bench_robbins_bvp_family(seed: int = _SEED + 30302) -> dict[str, float]:
    return _finite_blob(bench_robbins_bvp(seed))


def bench_bvp_eigen_family(seed: int = _SEED + 30303) -> dict[str, float]:
    return _finite_blob(bench_bvp_eigen(seed))


def bench_loser_tree_family(seed: int = _SEED + 30304) -> dict[str, float]:
    return _finite_blob(bench_loser_tree(seed))


def bench_fusion_tree_family(seed: int = _SEED + 30305) -> dict[str, float]:
    return _finite_blob(bench_fusion_tree(seed))
