"""Wave-1037 agriculture canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.agronomy import bench_agronomy
from quant_fund.models.animal_science import bench_animal_science
from quant_fund.models.crop_science import bench_crop_science
from quant_fund.models.horticulture import bench_horticulture
from quant_fund.models.pest_management import bench_pest_management
from quant_fund.models.soil_science import bench_soil_science

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


def bench_crop_science_family(seed: int = _SEED + 42500) -> dict[str, float]:
    return _finite_blob(bench_crop_science(seed))


def bench_soil_science_family(seed: int = _SEED + 42501) -> dict[str, float]:
    return _finite_blob(bench_soil_science(seed))


def bench_agronomy_family(seed: int = _SEED + 42502) -> dict[str, float]:
    return _finite_blob(bench_agronomy(seed))


def bench_animal_science_family(seed: int = _SEED + 42503) -> dict[str, float]:
    return _finite_blob(bench_animal_science(seed))


def bench_horticulture_family(seed: int = _SEED + 42504) -> dict[str, float]:
    return _finite_blob(bench_horticulture(seed))


def bench_pest_management_family(seed: int = _SEED + 42505) -> dict[str, float]:
    return _finite_blob(bench_pest_management(seed))
