"""Wave-1137 education-4 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.adult_education import bench_adult_education
from quant_fund.models.bilingual_education import bench_bilingual_education
from quant_fund.models.early_childhood_education import bench_early_childhood_education
from quant_fund.models.educational_leadership import bench_educational_leadership
from quant_fund.models.gifted_education import bench_gifted_education
from quant_fund.models.instructional_design import bench_instructional_design

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


def bench_early_childhood_education_family(seed: int = _SEED + 52500) -> dict[str, float]:
    return _finite_blob(bench_early_childhood_education(seed))


def bench_bilingual_education_family(seed: int = _SEED + 52501) -> dict[str, float]:
    return _finite_blob(bench_bilingual_education(seed))


def bench_gifted_education_family(seed: int = _SEED + 52502) -> dict[str, float]:
    return _finite_blob(bench_gifted_education(seed))


def bench_adult_education_family(seed: int = _SEED + 52503) -> dict[str, float]:
    return _finite_blob(bench_adult_education(seed))


def bench_instructional_design_family(seed: int = _SEED + 52504) -> dict[str, float]:
    return _finite_blob(bench_instructional_design(seed))


def bench_educational_leadership_family(seed: int = _SEED + 52505) -> dict[str, float]:
    return _finite_blob(bench_educational_leadership(seed))
