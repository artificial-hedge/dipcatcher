"""Wave-455 synthetic-math bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cubical_path import bench_cubical_path
from quant_fund.models.glue_types import bench_glue_types
from quant_fund.models.hcomp_fill import bench_hcomp_fill
from quant_fund.models.interval_obj import bench_interval_obj
from quant_fund.models.kan_op import bench_kan_op
from quant_fund.models.transport_coe import bench_transport_coe

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


def bench_cubical_path_family(seed: int = _SEED + 2636) -> dict[str, float]:
    return _floats(_finite_blob("cubical_path", bench_cubical_path(seed)))


def bench_hcomp_fill_family(seed: int = _SEED + 2637) -> dict[str, float]:
    return _floats(_finite_blob("hcomp_fill", bench_hcomp_fill(seed)))


def bench_glue_types_family(seed: int = _SEED + 2638) -> dict[str, float]:
    return _floats(_finite_blob("glue_types", bench_glue_types(seed)))


def bench_interval_obj_family(seed: int = _SEED + 2639) -> dict[str, float]:
    return _floats(_finite_blob("interval_obj", bench_interval_obj(seed)))


def bench_kan_op_family(seed: int = _SEED + 2640) -> dict[str, float]:
    return _floats(_finite_blob("kan_op", bench_kan_op(seed)))


def bench_transport_coe_family(seed: int = _SEED + 2641) -> dict[str, float]:
    return _floats(_finite_blob("transport_coe", bench_transport_coe(seed)))
