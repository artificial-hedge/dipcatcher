"""Wave-1106 materials-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.biomaterials import bench_biomaterials
from quant_fund.models.characterization_methods import bench_characterization_methods
from quant_fund.models.composite_materials import bench_composite_materials
from quant_fund.models.phase_diagrams import bench_phase_diagrams
from quant_fund.models.semiconductors_materials import bench_semiconductors_materials
from quant_fund.models.thin_films import bench_thin_films

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


def bench_semiconductors_materials_family(seed: int = _SEED + 49400) -> dict[str, float]:
    return _finite_blob(bench_semiconductors_materials(seed))


def bench_composite_materials_family(seed: int = _SEED + 49401) -> dict[str, float]:
    return _finite_blob(bench_composite_materials(seed))


def bench_thin_films_family(seed: int = _SEED + 49402) -> dict[str, float]:
    return _finite_blob(bench_thin_films(seed))


def bench_biomaterials_family(seed: int = _SEED + 49403) -> dict[str, float]:
    return _finite_blob(bench_biomaterials(seed))


def bench_phase_diagrams_family(seed: int = _SEED + 49404) -> dict[str, float]:
    return _finite_blob(bench_phase_diagrams(seed))


def bench_characterization_methods_family(seed: int = _SEED + 49405) -> dict[str, float]:
    return _finite_blob(bench_characterization_methods(seed))
