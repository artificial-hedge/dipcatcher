"""Wave-1075 architecture/design canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.architecture_theory import bench_architecture_theory
from quant_fund.models.building_science import bench_building_science
from quant_fund.models.industrial_design import bench_industrial_design
from quant_fund.models.interior_design import bench_interior_design
from quant_fund.models.landscape_architecture import bench_landscape_architecture
from quant_fund.models.urban_design import bench_urban_design

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


def bench_architecture_theory_family(seed: int = _SEED + 46300) -> dict[str, float]:
    return _finite_blob(bench_architecture_theory(seed))


def bench_urban_design_family(seed: int = _SEED + 46301) -> dict[str, float]:
    return _finite_blob(bench_urban_design(seed))


def bench_landscape_architecture_family(seed: int = _SEED + 46302) -> dict[str, float]:
    return _finite_blob(bench_landscape_architecture(seed))


def bench_interior_design_family(seed: int = _SEED + 46303) -> dict[str, float]:
    return _finite_blob(bench_interior_design(seed))


def bench_industrial_design_family(seed: int = _SEED + 46304) -> dict[str, float]:
    return _finite_blob(bench_industrial_design(seed))


def bench_building_science_family(seed: int = _SEED + 46305) -> dict[str, float]:
    return _finite_blob(bench_building_science(seed))
