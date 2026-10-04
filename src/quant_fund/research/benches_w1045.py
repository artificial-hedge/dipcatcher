"""Wave-1045 geology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.geochemistry import bench_geochemistry
from quant_fund.models.geochronology import bench_geochronology
from quant_fund.models.paleontology import bench_paleontology
from quant_fund.models.petrology import bench_petrology
from quant_fund.models.stratigraphy import bench_stratigraphy
from quant_fund.models.structural_geology import bench_structural_geology

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


def bench_stratigraphy_family(seed: int = _SEED + 43300) -> dict[str, float]:
    return _finite_blob(bench_stratigraphy(seed))


def bench_structural_geology_family(seed: int = _SEED + 43301) -> dict[str, float]:
    return _finite_blob(bench_structural_geology(seed))


def bench_petrology_family(seed: int = _SEED + 43302) -> dict[str, float]:
    return _finite_blob(bench_petrology(seed))


def bench_geochemistry_family(seed: int = _SEED + 43303) -> dict[str, float]:
    return _finite_blob(bench_geochemistry(seed))


def bench_geochronology_family(seed: int = _SEED + 43304) -> dict[str, float]:
    return _finite_blob(bench_geochronology(seed))


def bench_paleontology_family(seed: int = _SEED + 43305) -> dict[str, float]:
    return _finite_blob(bench_paleontology(seed))
