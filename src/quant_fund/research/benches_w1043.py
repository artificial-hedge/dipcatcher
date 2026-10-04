"""Wave-1043 forestry canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.dendrology import bench_dendrology
from quant_fund.models.forest_ecology import bench_forest_ecology
from quant_fund.models.forest_economics import bench_forest_economics
from quant_fund.models.silviculture import bench_silviculture
from quant_fund.models.timber_harvesting import bench_timber_harvesting
from quant_fund.models.wildfire_management import bench_wildfire_management

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


def bench_silviculture_family(seed: int = _SEED + 43100) -> dict[str, float]:
    return _finite_blob(bench_silviculture(seed))


def bench_forest_ecology_family(seed: int = _SEED + 43101) -> dict[str, float]:
    return _finite_blob(bench_forest_ecology(seed))


def bench_timber_harvesting_family(seed: int = _SEED + 43102) -> dict[str, float]:
    return _finite_blob(bench_timber_harvesting(seed))


def bench_forest_economics_family(seed: int = _SEED + 43103) -> dict[str, float]:
    return _finite_blob(bench_forest_economics(seed))


def bench_dendrology_family(seed: int = _SEED + 43104) -> dict[str, float]:
    return _finite_blob(bench_dendrology(seed))


def bench_wildfire_management_family(seed: int = _SEED + 43105) -> dict[str, float]:
    return _finite_blob(bench_wildfire_management(seed))
