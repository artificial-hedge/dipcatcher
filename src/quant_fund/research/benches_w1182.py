"""Wave-1182 dance canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.ballet_studies import bench_ballet_studies
from quant_fund.models.choreography_2 import bench_choreography_2
from quant_fund.models.dance_pedagogy import bench_dance_pedagogy
from quant_fund.models.dance_science import bench_dance_science
from quant_fund.models.movement_studies import bench_movement_studies
from quant_fund.models.somatic_practices import bench_somatic_practices

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


def bench_ballet_studies_family(seed: int = _SEED + 57000) -> dict[str, float]:
    return _finite_blob(bench_ballet_studies(seed))


def bench_choreography_2_family(seed: int = _SEED + 57001) -> dict[str, float]:
    return _finite_blob(bench_choreography_2(seed))


def bench_dance_pedagogy_family(seed: int = _SEED + 57002) -> dict[str, float]:
    return _finite_blob(bench_dance_pedagogy(seed))


def bench_somatic_practices_family(seed: int = _SEED + 57003) -> dict[str, float]:
    return _finite_blob(bench_somatic_practices(seed))


def bench_dance_science_family(seed: int = _SEED + 57004) -> dict[str, float]:
    return _finite_blob(bench_dance_science(seed))


def bench_movement_studies_family(seed: int = _SEED + 57005) -> dict[str, float]:
    return _finite_blob(bench_movement_studies(seed))
