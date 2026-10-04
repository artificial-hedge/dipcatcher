"""Wave-1148 geological-sciences canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.geochronology_2 import bench_geochronology_2
from quant_fund.models.geology_3 import bench_geology_3
from quant_fund.models.geomorphology_2 import bench_geomorphology_2
from quant_fund.models.mineralogy_2 import bench_mineralogy_2
from quant_fund.models.petrology_2 import bench_petrology_2
from quant_fund.models.stratigraphy_2 import bench_stratigraphy_2

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


def bench_geology_3_family(seed: int = _SEED + 53600) -> dict[str, float]:
    return _finite_blob(bench_geology_3(seed))


def bench_petrology_2_family(seed: int = _SEED + 53601) -> dict[str, float]:
    return _finite_blob(bench_petrology_2(seed))


def bench_mineralogy_2_family(seed: int = _SEED + 53602) -> dict[str, float]:
    return _finite_blob(bench_mineralogy_2(seed))


def bench_stratigraphy_2_family(seed: int = _SEED + 53603) -> dict[str, float]:
    return _finite_blob(bench_stratigraphy_2(seed))


def bench_geomorphology_2_family(seed: int = _SEED + 53604) -> dict[str, float]:
    return _finite_blob(bench_geomorphology_2(seed))


def bench_geochronology_2_family(seed: int = _SEED + 53605) -> dict[str, float]:
    return _finite_blob(bench_geochronology_2(seed))
