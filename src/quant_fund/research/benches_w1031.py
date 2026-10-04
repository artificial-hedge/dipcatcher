"""Wave-1031 civil-engineering canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.construction_mgmt import bench_construction_mgmt
from quant_fund.models.geotechnics import bench_geotechnics
from quant_fund.models.structural_analysis import bench_structural_analysis
from quant_fund.models.surveying import bench_surveying
from quant_fund.models.transportation_eng import bench_transportation_eng
from quant_fund.models.water_resources import bench_water_resources

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


def bench_structural_analysis_family(seed: int = _SEED + 41900) -> dict[str, float]:
    return _finite_blob(bench_structural_analysis(seed))


def bench_geotechnics_family(seed: int = _SEED + 41901) -> dict[str, float]:
    return _finite_blob(bench_geotechnics(seed))


def bench_transportation_eng_family(seed: int = _SEED + 41902) -> dict[str, float]:
    return _finite_blob(bench_transportation_eng(seed))


def bench_water_resources_family(seed: int = _SEED + 41903) -> dict[str, float]:
    return _finite_blob(bench_water_resources(seed))


def bench_construction_mgmt_family(seed: int = _SEED + 41904) -> dict[str, float]:
    return _finite_blob(bench_construction_mgmt(seed))


def bench_surveying_family(seed: int = _SEED + 41905) -> dict[str, float]:
    return _finite_blob(bench_surveying(seed))
