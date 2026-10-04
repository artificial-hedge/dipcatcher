"""Wave-415 topology-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.homeo_top import bench_homeo_top
from quant_fund.models.locally_compact import bench_locally_compact
from quant_fund.models.open_cover import bench_open_cover
from quant_fund.models.paracompact import bench_paracompact
from quant_fund.models.partition_unity import bench_partition_unity
from quant_fund.models.quotient_map import bench_quotient_map

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


def bench_quotient_map_family(
    seed: int = _SEED + 2396,
) -> dict[str, float]:
    return _floats(_finite_blob("quotient_map", bench_quotient_map(seed)))


def bench_open_cover_family(
    seed: int = _SEED + 2397,
) -> dict[str, float]:
    return _floats(_finite_blob("open_cover", bench_open_cover(seed)))


def bench_locally_compact_family(
    seed: int = _SEED + 2398,
) -> dict[str, float]:
    return _floats(_finite_blob("locally_compact", bench_locally_compact(seed)))


def bench_homeo_top_family(
    seed: int = _SEED + 2399,
) -> dict[str, float]:
    return _floats(_finite_blob("homeo_top", bench_homeo_top(seed)))


def bench_paracompact_family(
    seed: int = _SEED + 2400,
) -> dict[str, float]:
    return _floats(_finite_blob("paracompact", bench_paracompact(seed)))


def bench_partition_unity_family(
    seed: int = _SEED + 2401,
) -> dict[str, float]:
    return _floats(_finite_blob("partition_unity", bench_partition_unity(seed)))
