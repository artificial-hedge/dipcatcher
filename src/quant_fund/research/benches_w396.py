"""Wave-396 algebraic-number-theory-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cm_points import bench_cm_points
from quant_fund.models.cyclotomic_field import bench_cyclotomic_field
from quant_fund.models.hensel_field import bench_hensel_field
from quant_fund.models.idele_class import bench_idele_class
from quant_fund.models.kronecker_weber import bench_kronecker_weber
from quant_fund.models.local_field import bench_local_field

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


def bench_cyclotomic_field_family(
    seed: int = _SEED + 2282,
) -> dict[str, float]:
    return _floats(_finite_blob("cyclotomic_field", bench_cyclotomic_field(seed)))


def bench_kronecker_weber_family(
    seed: int = _SEED + 2283,
) -> dict[str, float]:
    return _floats(_finite_blob("kronecker_weber", bench_kronecker_weber(seed)))


def bench_local_field_family(seed: int = _SEED + 2284) -> dict[str, float]:
    return _floats(_finite_blob("local_field", bench_local_field(seed)))


def bench_hensel_field_family(seed: int = _SEED + 2285) -> dict[str, float]:
    return _floats(_finite_blob("hensel_field", bench_hensel_field(seed)))


def bench_cm_points_family(seed: int = _SEED + 2286) -> dict[str, float]:
    return _floats(_finite_blob("cm_points", bench_cm_points(seed)))


def bench_idele_class_family(seed: int = _SEED + 2287) -> dict[str, float]:
    return _floats(_finite_blob("idele_class", bench_idele_class(seed)))
