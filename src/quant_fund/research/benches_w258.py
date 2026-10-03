"""Wave-258 adapters: databases-3 canon — cascades optimizer,
vectorized execution, zone maps, FD discovery,
bitmap index, adaptive query — SYNTHETIC benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adaptive_qp import bench_adaptive_qp
from quant_fund.models.bitmap_index import bench_bitmap_index
from quant_fund.models.cascades_opt import bench_cascades_opt
from quant_fund.models.func_dep import bench_func_dep
from quant_fund.models.vectorized_exec import bench_vectorized_exec
from quant_fund.models.zone_map import bench_zone_map

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


def bench_cascades_opt_family(seed: int = _SEED + 1350) -> dict[str, float]:
    return bench_cascades_opt(seed)


def bench_vectorized_exec_family(seed: int = _SEED + 1351) -> dict[str, float]:
    return bench_vectorized_exec(seed)


def bench_zone_map_family(seed: int = _SEED + 1352) -> dict[str, float]:
    return bench_zone_map(seed)


def bench_func_dep_family(seed: int = _SEED + 1353) -> dict[str, float]:
    return bench_func_dep(seed)


def bench_bitmap_index_family(seed: int = _SEED + 1354) -> dict[str, float]:
    return bench_bitmap_index(seed)


def bench_adaptive_qp_family(seed: int = _SEED + 1355) -> dict[str, float]:
    return bench_adaptive_qp(seed)
