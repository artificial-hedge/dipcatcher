"""Wave-1200 medical-physics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.cardiovascular_technology import bench_cardiovascular_technology
from quant_fund.models.dosimetry_studies import bench_dosimetry_studies
from quant_fund.models.medical_physics_studies import bench_medical_physics_studies
from quant_fund.models.nuclear_medicine_technology import bench_nuclear_medicine_technology
from quant_fund.models.radiation_dosimetry import bench_radiation_dosimetry
from quant_fund.models.radiopharmacy import bench_radiopharmacy

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


def bench_cardiovascular_technology_family(seed: int = _SEED + 58800) -> dict[str, float]:
    return _finite_blob(bench_cardiovascular_technology(seed))


def bench_nuclear_medicine_technology_family(seed: int = _SEED + 58801) -> dict[str, float]:
    return _finite_blob(bench_nuclear_medicine_technology(seed))


def bench_radiation_dosimetry_family(seed: int = _SEED + 58802) -> dict[str, float]:
    return _finite_blob(bench_radiation_dosimetry(seed))


def bench_medical_physics_studies_family(seed: int = _SEED + 58803) -> dict[str, float]:
    return _finite_blob(bench_medical_physics_studies(seed))


def bench_dosimetry_studies_family(seed: int = _SEED + 58804) -> dict[str, float]:
    return _finite_blob(bench_dosimetry_studies(seed))


def bench_radiopharmacy_family(seed: int = _SEED + 58805) -> dict[str, float]:
    return _finite_blob(bench_radiopharmacy(seed))
