"""Wave-1060 education canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.assessment_theory import bench_assessment_theory
from quant_fund.models.curriculum_design import bench_curriculum_design
from quant_fund.models.educational_psychology import bench_educational_psychology
from quant_fund.models.educational_technology import bench_educational_technology
from quant_fund.models.learning_sciences import bench_learning_sciences
from quant_fund.models.pedagogy import bench_pedagogy

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


def bench_curriculum_design_family(seed: int = _SEED + 44800) -> dict[str, float]:
    return _finite_blob(bench_curriculum_design(seed))


def bench_pedagogy_family(seed: int = _SEED + 44801) -> dict[str, float]:
    return _finite_blob(bench_pedagogy(seed))


def bench_educational_psychology_family(seed: int = _SEED + 44802) -> dict[str, float]:
    return _finite_blob(bench_educational_psychology(seed))


def bench_assessment_theory_family(seed: int = _SEED + 44803) -> dict[str, float]:
    return _finite_blob(bench_assessment_theory(seed))


def bench_learning_sciences_family(seed: int = _SEED + 44804) -> dict[str, float]:
    return _finite_blob(bench_learning_sciences(seed))


def bench_educational_technology_family(seed: int = _SEED + 44805) -> dict[str, float]:
    return _finite_blob(bench_educational_technology(seed))
