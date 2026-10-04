"""Wave-1189 hospitality canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.event_management import bench_event_management
from quant_fund.models.hospitality_studies import bench_hospitality_studies
from quant_fund.models.hotel_management import bench_hotel_management
from quant_fund.models.leisure_science import bench_leisure_science
from quant_fund.models.recreation_management import bench_recreation_management
from quant_fund.models.tourism_studies import bench_tourism_studies

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


def bench_hospitality_studies_family(seed: int = _SEED + 57700) -> dict[str, float]:
    return _finite_blob(bench_hospitality_studies(seed))


def bench_event_management_family(seed: int = _SEED + 57701) -> dict[str, float]:
    return _finite_blob(bench_event_management(seed))


def bench_hotel_management_family(seed: int = _SEED + 57702) -> dict[str, float]:
    return _finite_blob(bench_hotel_management(seed))


def bench_tourism_studies_family(seed: int = _SEED + 57703) -> dict[str, float]:
    return _finite_blob(bench_tourism_studies(seed))


def bench_recreation_management_family(seed: int = _SEED + 57704) -> dict[str, float]:
    return _finite_blob(bench_recreation_management(seed))


def bench_leisure_science_family(seed: int = _SEED + 57705) -> dict[str, float]:
    return _finite_blob(bench_leisure_science(seed))
