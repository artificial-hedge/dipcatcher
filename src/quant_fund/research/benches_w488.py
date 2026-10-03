"""Wave-488 derived-geometry-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.d_critical import bench_d_critical
from quant_fund.models.derived_quot import bench_derived_quot
from quant_fund.models.intrinsic_be import bench_intrinsic_be
from quant_fund.models.perfect_obstruction import bench_perfect_obstruction
from quant_fund.models.shifted_tangent import bench_shifted_tangent
from quant_fund.models.virtual_pull import bench_virtual_pull

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


def bench_shifted_tangent_family(seed: int = _SEED + 2834) -> dict[str, float]:
    return _floats(_finite_blob("shifted_tangent", bench_shifted_tangent(seed)))


def bench_derived_quot_family(seed: int = _SEED + 2835) -> dict[str, float]:
    return _floats(_finite_blob("derived_quot", bench_derived_quot(seed)))


def bench_virtual_pull_family(seed: int = _SEED + 2836) -> dict[str, float]:
    return _floats(_finite_blob("virtual_pull", bench_virtual_pull(seed)))


def bench_intrinsic_be_family(seed: int = _SEED + 2837) -> dict[str, float]:
    return _floats(_finite_blob("intrinsic_be", bench_intrinsic_be(seed)))


def bench_d_critical_family(seed: int = _SEED + 2838) -> dict[str, float]:
    return _floats(_finite_blob("d_critical", bench_d_critical(seed)))


def bench_perfect_obstruction_family(seed: int = _SEED + 2839) -> dict[str, float]:
    return _floats(_finite_blob("perfect_obstruction", bench_perfect_obstruction(seed)))
