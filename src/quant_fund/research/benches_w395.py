"""Wave-395 combinatorial-enumeration bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bell_triangle import bench_bell_triangle
from quant_fund.models.catalan_dp import bench_catalan_dp
from quant_fund.models.eulerian_num import bench_eulerian_num
from quant_fund.models.inclusion_excl import bench_inclusion_excl
from quant_fund.models.partition_count import bench_partition_count
from quant_fund.models.stirling_cycle import bench_stirling_cycle

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


def bench_catalan_dp_family(seed: int = _SEED + 2276) -> dict[str, float]:
    return _floats(_finite_blob("catalan_dp", bench_catalan_dp(seed)))


def bench_stirling_cycle_family(seed: int = _SEED + 2277) -> dict[str, float]:
    return _floats(_finite_blob("stirling_cycle", bench_stirling_cycle(seed)))


def bench_partition_count_family(
    seed: int = _SEED + 2278,
) -> dict[str, float]:
    return _floats(_finite_blob("partition_count", bench_partition_count(seed)))


def bench_bell_triangle_family(seed: int = _SEED + 2279) -> dict[str, float]:
    return _floats(_finite_blob("bell_triangle", bench_bell_triangle(seed)))


def bench_eulerian_num_family(seed: int = _SEED + 2280) -> dict[str, float]:
    return _floats(_finite_blob("eulerian_num", bench_eulerian_num(seed)))


def bench_inclusion_excl_family(seed: int = _SEED + 2281) -> dict[str, float]:
    return _floats(_finite_blob("inclusion_excl", bench_inclusion_excl(seed)))
