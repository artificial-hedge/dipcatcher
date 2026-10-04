"""Wave-1078 performing arts canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.choreography import bench_choreography
from quant_fund.models.dance_studies import bench_dance_studies
from quant_fund.models.dramaturgy import bench_dramaturgy
from quant_fund.models.performance_theory import bench_performance_theory
from quant_fund.models.stage_design import bench_stage_design
from quant_fund.models.theater_studies import bench_theater_studies

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


def bench_theater_studies_family(seed: int = _SEED + 46600) -> dict[str, float]:
    return _finite_blob(bench_theater_studies(seed))


def bench_dance_studies_family(seed: int = _SEED + 46601) -> dict[str, float]:
    return _finite_blob(bench_dance_studies(seed))


def bench_performance_theory_family(seed: int = _SEED + 46602) -> dict[str, float]:
    return _finite_blob(bench_performance_theory(seed))


def bench_dramaturgy_family(seed: int = _SEED + 46603) -> dict[str, float]:
    return _finite_blob(bench_dramaturgy(seed))


def bench_choreography_family(seed: int = _SEED + 46604) -> dict[str, float]:
    return _finite_blob(bench_choreography(seed))


def bench_stage_design_family(seed: int = _SEED + 46605) -> dict[str, float]:
    return _finite_blob(bench_stage_design(seed))
