"""Wave-1036 petroleum-engineering canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.drilling_engineering import bench_drilling_engineering
from quant_fund.models.enhanced_recovery import bench_enhanced_recovery
from quant_fund.models.formation_evaluation import bench_formation_evaluation
from quant_fund.models.production_engineering import bench_production_engineering
from quant_fund.models.reservoir_engineering import bench_reservoir_engineering
from quant_fund.models.well_testing import bench_well_testing

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


def bench_reservoir_engineering_family(seed: int = _SEED + 42400) -> dict[str, float]:
    return _finite_blob(bench_reservoir_engineering(seed))


def bench_drilling_engineering_family(seed: int = _SEED + 42401) -> dict[str, float]:
    return _finite_blob(bench_drilling_engineering(seed))


def bench_production_engineering_family(seed: int = _SEED + 42402) -> dict[str, float]:
    return _finite_blob(bench_production_engineering(seed))


def bench_formation_evaluation_family(seed: int = _SEED + 42403) -> dict[str, float]:
    return _finite_blob(bench_formation_evaluation(seed))


def bench_well_testing_family(seed: int = _SEED + 42404) -> dict[str, float]:
    return _finite_blob(bench_well_testing(seed))


def bench_enhanced_recovery_family(seed: int = _SEED + 42405) -> dict[str, float]:
    return _finite_blob(bench_enhanced_recovery(seed))
