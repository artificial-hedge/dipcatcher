"""Wave-311 VLSI-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.aig_rewrite import bench_aig_rewrite
from quant_fund.models.clock_tree import bench_clock_tree
from quant_fund.models.floorplan_sa import bench_floorplan_sa
from quant_fund.models.fm_partition import bench_fm_partition
from quant_fund.models.lee_router import bench_lee_router
from quant_fund.models.power_est import bench_power_est

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


def bench_fm_partition_family(seed: int = _SEED + 1772) -> dict[str, float]:
    return _floats(_finite_blob("fm_partition", bench_fm_partition(seed)))


def bench_lee_router_family(seed: int = _SEED + 1773) -> dict[str, float]:
    return _floats(_finite_blob("lee_router", bench_lee_router(seed)))


def bench_clock_tree_family(seed: int = _SEED + 1774) -> dict[str, float]:
    return _floats(_finite_blob("clock_tree", bench_clock_tree(seed)))


def bench_aig_rewrite_family(seed: int = _SEED + 1775) -> dict[str, float]:
    return _floats(_finite_blob("aig_rewrite", bench_aig_rewrite(seed)))


def bench_power_est_family(seed: int = _SEED + 1776) -> dict[str, float]:
    return _floats(_finite_blob("power_est", bench_power_est(seed)))


def bench_floorplan_sa_family(seed: int = _SEED + 1777) -> dict[str, float]:
    return _floats(_finite_blob("floorplan_sa", bench_floorplan_sa(seed)))
