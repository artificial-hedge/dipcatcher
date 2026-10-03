"""Wave-769 Stein-method bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.barbour_stein import bench_barbour_stein
from quant_fund.models.chatt_stein import bench_chatt_stein
from quant_fund.models.chen_stein import bench_chen_stein
from quant_fund.models.ross_stein import bench_ross_stein
from quant_fund.models.stein_equation import (
    bench_stein_equation,
)
from quant_fund.models.stein_method import bench_stein_method

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


def bench_stein_method_family(
    seed: int = _SEED + 15800,
) -> dict[str, float]:
    return _floats(_finite_blob("stein_method", bench_stein_method(seed)))


def bench_stein_equation_family(
    seed: int = _SEED + 15801,
) -> dict[str, float]:
    return _floats(_finite_blob("stein_equation", bench_stein_equation(seed)))


def bench_barbour_stein_family(
    seed: int = _SEED + 15802,
) -> dict[str, float]:
    return _floats(_finite_blob("barbour_stein", bench_barbour_stein(seed)))


def bench_chen_stein_family(
    seed: int = _SEED + 15803,
) -> dict[str, float]:
    return _floats(_finite_blob("chen_stein", bench_chen_stein(seed)))


def bench_ross_stein_family(
    seed: int = _SEED + 15804,
) -> dict[str, float]:
    return _floats(_finite_blob("ross_stein", bench_ross_stein(seed)))


def bench_chatt_stein_family(
    seed: int = _SEED + 15805,
) -> dict[str, float]:
    return _floats(_finite_blob("chatt_stein", bench_chatt_stein(seed)))
