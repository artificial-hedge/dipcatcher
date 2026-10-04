"""Wave-1196 emergency-safety canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.disaster_management import bench_disaster_management
from quant_fund.models.emergency_medical_technician import bench_emergency_medical_technician
from quant_fund.models.fire_science_studies import bench_fire_science_studies
from quant_fund.models.industrial_hygiene import bench_industrial_hygiene
from quant_fund.models.occupational_safety import bench_occupational_safety
from quant_fund.models.paramedic_studies import bench_paramedic_studies

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


def bench_emergency_medical_technician_family(seed: int = _SEED + 58400) -> dict[str, float]:
    return _finite_blob(bench_emergency_medical_technician(seed))


def bench_fire_science_studies_family(seed: int = _SEED + 58401) -> dict[str, float]:
    return _finite_blob(bench_fire_science_studies(seed))


def bench_paramedic_studies_family(seed: int = _SEED + 58402) -> dict[str, float]:
    return _finite_blob(bench_paramedic_studies(seed))


def bench_disaster_management_family(seed: int = _SEED + 58403) -> dict[str, float]:
    return _finite_blob(bench_disaster_management(seed))


def bench_occupational_safety_family(seed: int = _SEED + 58404) -> dict[str, float]:
    return _finite_blob(bench_occupational_safety(seed))


def bench_industrial_hygiene_family(seed: int = _SEED + 58405) -> dict[str, float]:
    return _finite_blob(bench_industrial_hygiene(seed))
