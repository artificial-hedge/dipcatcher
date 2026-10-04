"""Wave-1211 neurology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.epilepsy_studies import bench_epilepsy_studies
from quant_fund.models.headache_medicine import bench_headache_medicine
from quant_fund.models.movement_disorders import bench_movement_disorders
from quant_fund.models.neurodevelopmental_disorders import bench_neurodevelopmental_disorders
from quant_fund.models.neuropsychiatry_studies import bench_neuropsychiatry_studies
from quant_fund.models.pediatric_neurology import bench_pediatric_neurology

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


def bench_pediatric_neurology_family(seed: int = _SEED + 59900) -> dict[str, float]:
    return _finite_blob(bench_pediatric_neurology(seed))


def bench_neurodevelopmental_disorders_family(seed: int = _SEED + 59901) -> dict[str, float]:
    return _finite_blob(bench_neurodevelopmental_disorders(seed))


def bench_neuropsychiatry_studies_family(seed: int = _SEED + 59902) -> dict[str, float]:
    return _finite_blob(bench_neuropsychiatry_studies(seed))


def bench_headache_medicine_family(seed: int = _SEED + 59903) -> dict[str, float]:
    return _finite_blob(bench_headache_medicine(seed))


def bench_epilepsy_studies_family(seed: int = _SEED + 59904) -> dict[str, float]:
    return _finite_blob(bench_epilepsy_studies(seed))


def bench_movement_disorders_family(seed: int = _SEED + 59905) -> dict[str, float]:
    return _finite_blob(bench_movement_disorders(seed))
