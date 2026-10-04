"""Wave-1230 emergency-medicine canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.acute_care_studies import bench_acute_care_studies
from quant_fund.models.disaster_medicine import bench_disaster_medicine
from quant_fund.models.emergency_medicine_studies import bench_emergency_medicine_studies
from quant_fund.models.resuscitation_medicine import bench_resuscitation_medicine
from quant_fund.models.toxicology_medicine import bench_toxicology_medicine
from quant_fund.models.trauma_medicine import bench_trauma_medicine

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


def bench_emergency_medicine_studies_family(seed: int = _SEED + 61800) -> dict[str, float]:
    return _finite_blob(bench_emergency_medicine_studies(seed))


def bench_trauma_medicine_family(seed: int = _SEED + 61801) -> dict[str, float]:
    return _finite_blob(bench_trauma_medicine(seed))


def bench_toxicology_medicine_family(seed: int = _SEED + 61802) -> dict[str, float]:
    return _finite_blob(bench_toxicology_medicine(seed))


def bench_disaster_medicine_family(seed: int = _SEED + 61803) -> dict[str, float]:
    return _finite_blob(bench_disaster_medicine(seed))


def bench_acute_care_studies_family(seed: int = _SEED + 61804) -> dict[str, float]:
    return _finite_blob(bench_acute_care_studies(seed))


def bench_resuscitation_medicine_family(seed: int = _SEED + 61805) -> dict[str, float]:
    return _finite_blob(bench_resuscitation_medicine(seed))
