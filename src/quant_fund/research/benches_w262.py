"""Wave-262 HPC canon adapter: SYNTHETIC benches."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.mesi_cache import bench_mesi_cache
from quant_fund.models.numa_alloc import bench_numa_alloc
from quant_fund.models.ring_allreduce import bench_ring_allreduce
from quant_fund.models.simd_lanes import bench_simd_lanes
from quant_fund.models.stencil_halo import bench_stencil_halo
from quant_fund.models.task_dag import bench_task_dag

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


def bench_stencil_halo_family(seed: int = _SEED + 1390) -> dict[str, float]:
    return bench_stencil_halo(seed)


def bench_mesi_cache_family(seed: int = _SEED + 1391) -> dict[str, float]:
    return bench_mesi_cache(seed)


def bench_ring_allreduce_family(seed: int = _SEED + 1392) -> dict[str, float]:
    return bench_ring_allreduce(seed)


def bench_simd_lanes_family(seed: int = _SEED + 1393) -> dict[str, float]:
    return bench_simd_lanes(seed)


def bench_task_dag_family(seed: int = _SEED + 1394) -> dict[str, float]:
    return bench_task_dag(seed)


def bench_numa_alloc_family(seed: int = _SEED + 1395) -> dict[str, float]:
    return bench_numa_alloc(seed)
