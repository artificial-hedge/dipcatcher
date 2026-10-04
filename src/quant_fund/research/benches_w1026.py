"""Wave-1026 environmental-science canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.atmospheric_chem import bench_atmospheric_chem
from quant_fund.models.carbon_cycle import bench_carbon_cycle
from quant_fund.models.climate_model import bench_climate_model
from quant_fund.models.ecosystem_model import bench_ecosystem_model
from quant_fund.models.hydrology import bench_hydrology
from quant_fund.models.ocean_circulation import bench_ocean_circulation

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


def bench_climate_model_family(seed: int = _SEED + 41400) -> dict[str, float]:
    return _finite_blob(bench_climate_model(seed))


def bench_ocean_circulation_family(seed: int = _SEED + 41401) -> dict[str, float]:
    return _finite_blob(bench_ocean_circulation(seed))


def bench_atmospheric_chem_family(seed: int = _SEED + 41402) -> dict[str, float]:
    return _finite_blob(bench_atmospheric_chem(seed))


def bench_hydrology_family(seed: int = _SEED + 41403) -> dict[str, float]:
    return _finite_blob(bench_hydrology(seed))


def bench_carbon_cycle_family(seed: int = _SEED + 41404) -> dict[str, float]:
    return _finite_blob(bench_carbon_cycle(seed))


def bench_ecosystem_model_family(seed: int = _SEED + 41405) -> dict[str, float]:
    return _finite_blob(bench_ecosystem_model(seed))
