"""Wave-779 matrix-analytic bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.logarithmic_red import (
    bench_logarithmic_red,
)
from quant_fund.models.matrix_geom import bench_matrix_geom
from quant_fund.models.neuts_map import bench_neuts_map
from quant_fund.models.phase_type import bench_phase_type
from quant_fund.models.quasi_birth import bench_quasi_birth
from quant_fund.models.ramaswami import bench_ramaswami

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


def bench_neuts_map_family(
    seed: int = _SEED + 16800,
) -> dict[str, float]:
    return _floats(_finite_blob("neuts_map", bench_neuts_map(seed)))


def bench_phase_type_family(
    seed: int = _SEED + 16801,
) -> dict[str, float]:
    return _floats(_finite_blob("phase_type", bench_phase_type(seed)))


def bench_matrix_geom_family(
    seed: int = _SEED + 16802,
) -> dict[str, float]:
    return _floats(_finite_blob("matrix_geom", bench_matrix_geom(seed)))


def bench_quasi_birth_family(
    seed: int = _SEED + 16803,
) -> dict[str, float]:
    return _floats(_finite_blob("quasi_birth", bench_quasi_birth(seed)))


def bench_ramaswami_family(
    seed: int = _SEED + 16804,
) -> dict[str, float]:
    return _floats(_finite_blob("ramaswami", bench_ramaswami(seed)))


def bench_logarithmic_red_family(
    seed: int = _SEED + 16805,
) -> dict[str, float]:
    return _floats(_finite_blob("logarithmic_red", bench_logarithmic_red(seed)))
