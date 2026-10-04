"""Wave-1065 geography canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.cartography import bench_cartography
from quant_fund.models.climatology import bench_climatology
from quant_fund.models.geomorphology import bench_geomorphology
from quant_fund.models.human_geography import bench_human_geography
from quant_fund.models.physical_geography import bench_physical_geography
from quant_fund.models.remote_sensing import bench_remote_sensing

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


def bench_physical_geography_family(seed: int = _SEED + 45300) -> dict[str, float]:
    return _finite_blob(bench_physical_geography(seed))


def bench_human_geography_family(seed: int = _SEED + 45301) -> dict[str, float]:
    return _finite_blob(bench_human_geography(seed))


def bench_cartography_family(seed: int = _SEED + 45302) -> dict[str, float]:
    return _finite_blob(bench_cartography(seed))


def bench_remote_sensing_family(seed: int = _SEED + 45303) -> dict[str, float]:
    return _finite_blob(bench_remote_sensing(seed))


def bench_geomorphology_family(seed: int = _SEED + 45304) -> dict[str, float]:
    return _finite_blob(bench_geomorphology(seed))


def bench_climatology_family(seed: int = _SEED + 45305) -> dict[str, float]:
    return _finite_blob(bench_climatology(seed))
