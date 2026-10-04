"""Wave-1096 environmental-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.conservation_biology import bench_conservation_biology
from quant_fund.models.environmental_toxicology import bench_environmental_toxicology
from quant_fund.models.landscape_ecology import bench_landscape_ecology
from quant_fund.models.marine_conservation import bench_marine_conservation
from quant_fund.models.pollution_science import bench_pollution_science
from quant_fund.models.urban_ecology import bench_urban_ecology

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


def bench_pollution_science_family(seed: int = _SEED + 48400) -> dict[str, float]:
    return _finite_blob(bench_pollution_science(seed))


def bench_conservation_biology_family(seed: int = _SEED + 48401) -> dict[str, float]:
    return _finite_blob(bench_conservation_biology(seed))


def bench_environmental_toxicology_family(seed: int = _SEED + 48402) -> dict[str, float]:
    return _finite_blob(bench_environmental_toxicology(seed))


def bench_urban_ecology_family(seed: int = _SEED + 48403) -> dict[str, float]:
    return _finite_blob(bench_urban_ecology(seed))


def bench_landscape_ecology_family(seed: int = _SEED + 48404) -> dict[str, float]:
    return _finite_blob(bench_landscape_ecology(seed))


def bench_marine_conservation_family(seed: int = _SEED + 48405) -> dict[str, float]:
    return _finite_blob(bench_marine_conservation(seed))
