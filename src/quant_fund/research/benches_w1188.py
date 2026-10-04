"""Wave-1188 culinary canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.brewing_science import bench_brewing_science
from quant_fund.models.culinary_science import bench_culinary_science
from quant_fund.models.enology import bench_enology
from quant_fund.models.fermentation_studies import bench_fermentation_studies
from quant_fund.models.gastronomy_2 import bench_gastronomy_2
from quant_fund.models.pastry_arts import bench_pastry_arts

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


def bench_culinary_science_family(seed: int = _SEED + 57600) -> dict[str, float]:
    return _finite_blob(bench_culinary_science(seed))


def bench_pastry_arts_family(seed: int = _SEED + 57601) -> dict[str, float]:
    return _finite_blob(bench_pastry_arts(seed))


def bench_brewing_science_family(seed: int = _SEED + 57602) -> dict[str, float]:
    return _finite_blob(bench_brewing_science(seed))


def bench_enology_family(seed: int = _SEED + 57603) -> dict[str, float]:
    return _finite_blob(bench_enology(seed))


def bench_fermentation_studies_family(seed: int = _SEED + 57604) -> dict[str, float]:
    return _finite_blob(bench_fermentation_studies(seed))


def bench_gastronomy_2_family(seed: int = _SEED + 57605) -> dict[str, float]:
    return _finite_blob(bench_gastronomy_2(seed))
