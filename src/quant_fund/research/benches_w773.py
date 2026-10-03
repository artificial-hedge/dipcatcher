"""Wave-773 queueing bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bulk_queue import bench_bulk_queue
from quant_fund.models.gm_queue import bench_gm_queue
from quant_fund.models.mg1_queue import bench_mg1_queue
from quant_fund.models.mm1_queue import bench_mm1_queue
from quant_fund.models.priority_queue import bench_priority_queue
from quant_fund.models.retrial_queue import bench_retrial_queue

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


def bench_mm1_queue_family(
    seed: int = _SEED + 16200,
) -> dict[str, float]:
    return _floats(_finite_blob("mm1_queue", bench_mm1_queue(seed)))


def bench_mg1_queue_family(
    seed: int = _SEED + 16201,
) -> dict[str, float]:
    return _floats(_finite_blob("mg1_queue", bench_mg1_queue(seed)))


def bench_gm_queue_family(
    seed: int = _SEED + 16202,
) -> dict[str, float]:
    return _floats(_finite_blob("gm_queue", bench_gm_queue(seed)))


def bench_bulk_queue_family(
    seed: int = _SEED + 16203,
) -> dict[str, float]:
    return _floats(_finite_blob("bulk_queue", bench_bulk_queue(seed)))


def bench_retrial_queue_family(
    seed: int = _SEED + 16204,
) -> dict[str, float]:
    return _floats(_finite_blob("retrial_queue", bench_retrial_queue(seed)))


def bench_priority_queue_family(
    seed: int = _SEED + 16205,
) -> dict[str, float]:
    return _floats(_finite_blob("priority_queue", bench_priority_queue(seed)))
