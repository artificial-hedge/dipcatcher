"""Wave-1091 education-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.comparative_education import bench_comparative_education
from quant_fund.models.distance_learning import bench_distance_learning
from quant_fund.models.higher_education import bench_higher_education
from quant_fund.models.literacy_studies import bench_literacy_studies
from quant_fund.models.special_education import bench_special_education
from quant_fund.models.vocational_education import bench_vocational_education

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


def bench_higher_education_family(seed: int = _SEED + 47900) -> dict[str, float]:
    return _finite_blob(bench_higher_education(seed))


def bench_vocational_education_family(seed: int = _SEED + 47901) -> dict[str, float]:
    return _finite_blob(bench_vocational_education(seed))


def bench_special_education_family(seed: int = _SEED + 47902) -> dict[str, float]:
    return _finite_blob(bench_special_education(seed))


def bench_comparative_education_family(seed: int = _SEED + 47903) -> dict[str, float]:
    return _finite_blob(bench_comparative_education(seed))


def bench_literacy_studies_family(seed: int = _SEED + 47904) -> dict[str, float]:
    return _finite_blob(bench_literacy_studies(seed))


def bench_distance_learning_family(seed: int = _SEED + 47905) -> dict[str, float]:
    return _finite_blob(bench_distance_learning(seed))
