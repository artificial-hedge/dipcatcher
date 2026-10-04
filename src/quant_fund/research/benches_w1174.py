"""Wave-1174 recreation canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.hospitality import bench_hospitality
from quant_fund.models.leisure_studies import bench_leisure_studies
from quant_fund.models.recreation import bench_recreation
from quant_fund.models.recreation_therapy import bench_recreation_therapy
from quant_fund.models.sports_management import bench_sports_management
from quant_fund.models.tourism import bench_tourism

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


def bench_recreation_family(seed: int = _SEED + 56200) -> dict[str, float]:
    return _finite_blob(bench_recreation(seed))


def bench_leisure_studies_family(seed: int = _SEED + 56201) -> dict[str, float]:
    return _finite_blob(bench_leisure_studies(seed))


def bench_tourism_family(seed: int = _SEED + 56202) -> dict[str, float]:
    return _finite_blob(bench_tourism(seed))


def bench_hospitality_family(seed: int = _SEED + 56203) -> dict[str, float]:
    return _finite_blob(bench_hospitality(seed))


def bench_sports_management_family(seed: int = _SEED + 56204) -> dict[str, float]:
    return _finite_blob(bench_sports_management(seed))


def bench_recreation_therapy_family(seed: int = _SEED + 56205) -> dict[str, float]:
    return _finite_blob(bench_recreation_therapy(seed))
