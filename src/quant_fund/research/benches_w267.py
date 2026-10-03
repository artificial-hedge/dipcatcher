"""Wave-267 GPU-architecture benches: scheduling, divergence, memory subsystem."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bank_conflict import bench_bank_conflict
from quant_fund.models.mem_coalesce import bench_mem_coalesce
from quant_fund.models.occupancy_calc import bench_occupancy_calc
from quant_fund.models.shared_mem_tile import bench_shared_mem_tile
from quant_fund.models.simt_divergence import bench_simt_divergence
from quant_fund.models.warp_scheduler import bench_warp_scheduler

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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


def bench_warp_scheduler_family(seed: int = _SEED + 1440) -> dict[str, float]:
    return _floats(_finite_blob("warp_scheduler", bench_warp_scheduler(seed)))


def bench_simt_divergence_family(seed: int = _SEED + 1441) -> dict[str, float]:
    return _floats(_finite_blob("simt_divergence", bench_simt_divergence(seed)))


def bench_bank_conflict_family(seed: int = _SEED + 1442) -> dict[str, float]:
    return _floats(_finite_blob("bank_conflict", bench_bank_conflict(seed)))


def bench_mem_coalesce_family(seed: int = _SEED + 1443) -> dict[str, float]:
    return _floats(_finite_blob("mem_coalesce", bench_mem_coalesce(seed)))


def bench_occupancy_calc_family(seed: int = _SEED + 1444) -> dict[str, float]:
    return _floats(_finite_blob("occupancy_calc", bench_occupancy_calc(seed)))


def bench_shared_mem_tile_family(seed: int = _SEED + 1445) -> dict[str, float]:
    return _floats(_finite_blob("shared_mem_tile", bench_shared_mem_tile(seed)))
