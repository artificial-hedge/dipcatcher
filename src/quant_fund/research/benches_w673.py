"""Wave-673 derived-geometry-6 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.derived_cohom import bench_derived_cohom
from quant_fund.models.derived_fiber2 import bench_derived_fiber2
from quant_fund.models.derived_intersection import (
    bench_derived_intersection,
)
from quant_fund.models.relative_trace import bench_relative_trace
from quant_fund.models.spectral_deformation2 import (
    bench_spectral_deformation2,
)
from quant_fund.models.virtual_class2 import bench_virtual_class2

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


def bench_derived_cohom_family(
    seed: int = _SEED + 6200,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_cohom", bench_derived_cohom(seed)))


def bench_spectral_deformation2_family(
    seed: int = _SEED + 6201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_deformation2",
            bench_spectral_deformation2(seed),
        )
    )


def bench_virtual_class2_family(
    seed: int = _SEED + 6202,
) -> dict[str, float]:
    return _floats(_finite_blob("virtual_class2", bench_virtual_class2(seed)))


def bench_derived_intersection_family(
    seed: int = _SEED + 6203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_intersection",
            bench_derived_intersection(seed),
        )
    )


def bench_derived_fiber2_family(
    seed: int = _SEED + 6204,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_fiber2", bench_derived_fiber2(seed)))


def bench_relative_trace_family(
    seed: int = _SEED + 6205,
) -> dict[str, float]:
    return _floats(_finite_blob("relative_trace", bench_relative_trace(seed)))
