"""Wave-1035 nuclear-engineering canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.isotope_production import bench_isotope_production
from quant_fund.models.nuclear_fuel_cycle import bench_nuclear_fuel_cycle
from quant_fund.models.nuclear_safety import bench_nuclear_safety
from quant_fund.models.radiation_protection import bench_radiation_protection
from quant_fund.models.reactor_physics import bench_reactor_physics
from quant_fund.models.thermal_hydraulics import bench_thermal_hydraulics

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


def bench_reactor_physics_family(seed: int = _SEED + 42300) -> dict[str, float]:
    return _finite_blob(bench_reactor_physics(seed))


def bench_radiation_protection_family(seed: int = _SEED + 42301) -> dict[str, float]:
    return _finite_blob(bench_radiation_protection(seed))


def bench_nuclear_fuel_cycle_family(seed: int = _SEED + 42302) -> dict[str, float]:
    return _finite_blob(bench_nuclear_fuel_cycle(seed))


def bench_thermal_hydraulics_family(seed: int = _SEED + 42303) -> dict[str, float]:
    return _finite_blob(bench_thermal_hydraulics(seed))


def bench_nuclear_safety_family(seed: int = _SEED + 42304) -> dict[str, float]:
    return _finite_blob(bench_nuclear_safety(seed))


def bench_isotope_production_family(seed: int = _SEED + 42305) -> dict[str, float]:
    return _finite_blob(bench_isotope_production(seed))
