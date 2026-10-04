"""Wave-752 UST/LERW bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.benjamini_ust import bench_benjamini_ust
from quant_fund.models.kirchhoff_matrix import (
    bench_kirchhoff_matrix,
)
from quant_fund.models.lawler_lerw import bench_lawler_lerw
from quant_fund.models.pemantle_ust import bench_pemantle_ust
from quant_fund.models.schramm_lerw import bench_schramm_lerw
from quant_fund.models.wilson_ust import bench_wilson_ust

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


def bench_wilson_ust_family(
    seed: int = _SEED + 14100,
) -> dict[str, float]:
    return _floats(_finite_blob("wilson_ust", bench_wilson_ust(seed)))


def bench_lawler_lerw_family(
    seed: int = _SEED + 14101,
) -> dict[str, float]:
    return _floats(_finite_blob("lawler_lerw", bench_lawler_lerw(seed)))


def bench_benjamini_ust_family(
    seed: int = _SEED + 14102,
) -> dict[str, float]:
    return _floats(_finite_blob("benjamini_ust", bench_benjamini_ust(seed)))


def bench_kirchhoff_matrix_family(
    seed: int = _SEED + 14103,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kirchhoff_matrix",
            bench_kirchhoff_matrix(seed),
        )
    )


def bench_pemantle_ust_family(
    seed: int = _SEED + 14104,
) -> dict[str, float]:
    return _floats(_finite_blob("pemantle_ust", bench_pemantle_ust(seed)))


def bench_schramm_lerw_family(
    seed: int = _SEED + 14105,
) -> dict[str, float]:
    return _floats(_finite_blob("schramm_lerw", bench_schramm_lerw(seed)))
