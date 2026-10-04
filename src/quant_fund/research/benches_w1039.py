"""Wave-1039 environmental-engineering canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.air_pollution_control import bench_air_pollution_control
from quant_fund.models.environmental_remediation import bench_environmental_remediation
from quant_fund.models.noise_control import bench_noise_control
from quant_fund.models.waste_management import bench_waste_management
from quant_fund.models.wastewater_engineering import bench_wastewater_engineering
from quant_fund.models.water_treatment import bench_water_treatment

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


def bench_water_treatment_family(seed: int = _SEED + 42700) -> dict[str, float]:
    return _finite_blob(bench_water_treatment(seed))


def bench_air_pollution_control_family(seed: int = _SEED + 42701) -> dict[str, float]:
    return _finite_blob(bench_air_pollution_control(seed))


def bench_waste_management_family(seed: int = _SEED + 42702) -> dict[str, float]:
    return _finite_blob(bench_waste_management(seed))


def bench_environmental_remediation_family(seed: int = _SEED + 42703) -> dict[str, float]:
    return _finite_blob(bench_environmental_remediation(seed))


def bench_wastewater_engineering_family(seed: int = _SEED + 42704) -> dict[str, float]:
    return _finite_blob(bench_wastewater_engineering(seed))


def bench_noise_control_family(seed: int = _SEED + 42705) -> dict[str, float]:
    return _finite_blob(bench_noise_control(seed))
