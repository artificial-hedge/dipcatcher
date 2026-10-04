"""Wave-1068 military/defense studies canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.conflict_resolution import bench_conflict_resolution
from quant_fund.models.defense_studies import bench_defense_studies
from quant_fund.models.intelligence_studies import bench_intelligence_studies
from quant_fund.models.military_science import bench_military_science
from quant_fund.models.peace_studies import bench_peace_studies
from quant_fund.models.strategic_studies import bench_strategic_studies

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


def bench_military_science_family(seed: int = _SEED + 45600) -> dict[str, float]:
    return _finite_blob(bench_military_science(seed))


def bench_defense_studies_family(seed: int = _SEED + 45601) -> dict[str, float]:
    return _finite_blob(bench_defense_studies(seed))


def bench_strategic_studies_family(seed: int = _SEED + 45602) -> dict[str, float]:
    return _finite_blob(bench_strategic_studies(seed))


def bench_intelligence_studies_family(seed: int = _SEED + 45603) -> dict[str, float]:
    return _finite_blob(bench_intelligence_studies(seed))


def bench_peace_studies_family(seed: int = _SEED + 45604) -> dict[str, float]:
    return _finite_blob(bench_peace_studies(seed))


def bench_conflict_resolution_family(seed: int = _SEED + 45605) -> dict[str, float]:
    return _finite_blob(bench_conflict_resolution(seed))
