"""Wave-970 subfactor-theory canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.fusion_algebra import bench_fusion_algebra
from quant_fund.models.paragroup import bench_paragroup
from quant_fund.models.planar_algebra import bench_planar_algebra
from quant_fund.models.principal_graph import bench_principal_graph
from quant_fund.models.standard_invariant import bench_standard_invariant
from quant_fund.models.subfactor import bench_subfactor

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


def bench_subfactor_family(seed: int = _SEED + 35800) -> dict[str, float]:
    return _finite_blob(bench_subfactor(seed))


def bench_standard_invariant_family(seed: int = _SEED + 35801) -> dict[str, float]:
    return _finite_blob(bench_standard_invariant(seed))


def bench_planar_algebra_family(seed: int = _SEED + 35802) -> dict[str, float]:
    return _finite_blob(bench_planar_algebra(seed))


def bench_paragroup_family(seed: int = _SEED + 35803) -> dict[str, float]:
    return _finite_blob(bench_paragroup(seed))


def bench_principal_graph_family(seed: int = _SEED + 35804) -> dict[str, float]:
    return _finite_blob(bench_principal_graph(seed))


def bench_fusion_algebra_family(seed: int = _SEED + 35805) -> dict[str, float]:
    return _finite_blob(bench_fusion_algebra(seed))
