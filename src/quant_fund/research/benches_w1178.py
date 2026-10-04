"""Wave-1178 interdisciplinary canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.cognitive_science_2 import bench_cognitive_science_2
from quant_fund.models.complexity_science import bench_complexity_science
from quant_fund.models.futures_studies import bench_futures_studies
from quant_fund.models.human_computer_interaction import bench_human_computer_interaction
from quant_fund.models.interdisciplinary_studies import bench_interdisciplinary_studies
from quant_fund.models.systems_science import bench_systems_science

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


def bench_interdisciplinary_studies_family(seed: int = _SEED + 56600) -> dict[str, float]:
    return _finite_blob(bench_interdisciplinary_studies(seed))


def bench_cognitive_science_2_family(seed: int = _SEED + 56601) -> dict[str, float]:
    return _finite_blob(bench_cognitive_science_2(seed))


def bench_futures_studies_family(seed: int = _SEED + 56602) -> dict[str, float]:
    return _finite_blob(bench_futures_studies(seed))


def bench_complexity_science_family(seed: int = _SEED + 56603) -> dict[str, float]:
    return _finite_blob(bench_complexity_science(seed))


def bench_systems_science_family(seed: int = _SEED + 56604) -> dict[str, float]:
    return _finite_blob(bench_systems_science(seed))


def bench_human_computer_interaction_family(seed: int = _SEED + 56605) -> dict[str, float]:
    return _finite_blob(bench_human_computer_interaction(seed))
