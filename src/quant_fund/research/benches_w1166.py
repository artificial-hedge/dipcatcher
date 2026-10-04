"""Wave-1166 design canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.architecture_2 import bench_architecture_2
from quant_fund.models.graphic_design_2 import bench_graphic_design_2
from quant_fund.models.industrial_design_2 import bench_industrial_design_2
from quant_fund.models.interior_design_2 import bench_interior_design_2
from quant_fund.models.landscape_architecture_2 import bench_landscape_architecture_2
from quant_fund.models.urban_planning_2 import bench_urban_planning_2

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


def bench_architecture_2_family(seed: int = _SEED + 55400) -> dict[str, float]:
    return _finite_blob(bench_architecture_2(seed))


def bench_urban_planning_2_family(seed: int = _SEED + 55401) -> dict[str, float]:
    return _finite_blob(bench_urban_planning_2(seed))


def bench_interior_design_2_family(seed: int = _SEED + 55402) -> dict[str, float]:
    return _finite_blob(bench_interior_design_2(seed))


def bench_landscape_architecture_2_family(seed: int = _SEED + 55403) -> dict[str, float]:
    return _finite_blob(bench_landscape_architecture_2(seed))


def bench_industrial_design_2_family(seed: int = _SEED + 55404) -> dict[str, float]:
    return _finite_blob(bench_industrial_design_2(seed))


def bench_graphic_design_2_family(seed: int = _SEED + 55405) -> dict[str, float]:
    return _finite_blob(bench_graphic_design_2(seed))
