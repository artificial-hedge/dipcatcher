"""Wave-1051 public-health canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.biostatistics_2 import bench_biostatistics_2
from quant_fund.models.epidemiology_2 import bench_epidemiology_2
from quant_fund.models.global_health import bench_global_health
from quant_fund.models.health_policy import bench_health_policy
from quant_fund.models.occupational_health import bench_occupational_health
from quant_fund.models.preventive_medicine import bench_preventive_medicine

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in blob.items():
        if key.lower() in _FORBIDDEN:
            raise ValueError(f"forbidden metric key: {key}")
        if not math.isfinite(val):
            raise ValueError(f"non-finite metric: {key}")
        out[key] = float(val)
    return out


def _floats(xs: Iterable[float]) -> list[float]:
    return [float(x) for x in xs]


def bench_epidemiology_2_family(seed: int = _SEED + 43900) -> dict[str, float]:
    return _finite_blob(bench_epidemiology_2(seed))


def bench_biostatistics_2_family(seed: int = _SEED + 43901) -> dict[str, float]:
    return _finite_blob(bench_biostatistics_2(seed))


def bench_health_policy_family(seed: int = _SEED + 43902) -> dict[str, float]:
    return _finite_blob(bench_health_policy(seed))


def bench_global_health_family(seed: int = _SEED + 43903) -> dict[str, float]:
    return _finite_blob(bench_global_health(seed))


def bench_occupational_health_family(seed: int = _SEED + 43904) -> dict[str, float]:
    return _finite_blob(bench_occupational_health(seed))


def bench_preventive_medicine_family(seed: int = _SEED + 43905) -> dict[str, float]:
    return _finite_blob(bench_preventive_medicine(seed))
