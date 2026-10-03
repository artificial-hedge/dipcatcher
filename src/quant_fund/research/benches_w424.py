"""Wave-424 set-theory-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.closed_unbounded import bench_closed_unbounded
from quant_fund.models.club_set import bench_club_set
from quant_fund.models.mahlo_cardinal import bench_mahlo_cardinal
from quant_fund.models.partition_calc import bench_partition_calc
from quant_fund.models.stationary_set import bench_stationary_set
from quant_fund.models.ultrafilter_toy import bench_ultrafilter_toy

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


def bench_club_set_family(
    seed: int = _SEED + 2450,
) -> dict[str, float]:
    return _floats(_finite_blob("club_set", bench_club_set(seed)))


def bench_stationary_set_family(
    seed: int = _SEED + 2451,
) -> dict[str, float]:
    return _floats(_finite_blob("stationary_set", bench_stationary_set(seed)))


def bench_ultrafilter_toy_family(
    seed: int = _SEED + 2452,
) -> dict[str, float]:
    return _floats(_finite_blob("ultrafilter_toy", bench_ultrafilter_toy(seed)))


def bench_partition_calc_family(
    seed: int = _SEED + 2453,
) -> dict[str, float]:
    return _floats(_finite_blob("partition_calc", bench_partition_calc(seed)))


def bench_closed_unbounded_family(
    seed: int = _SEED + 2454,
) -> dict[str, float]:
    return _floats(_finite_blob("closed_unbounded", bench_closed_unbounded(seed)))


def bench_mahlo_cardinal_family(
    seed: int = _SEED + 2455,
) -> dict[str, float]:
    return _floats(_finite_blob("mahlo_cardinal", bench_mahlo_cardinal(seed)))
