"""Wave-1052 nutrition canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.clinical_nutrition import bench_clinical_nutrition
from quant_fund.models.dietary_assessment import bench_dietary_assessment
from quant_fund.models.metabolic_health import bench_metabolic_health
from quant_fund.models.nutritional_biochemistry import bench_nutritional_biochemistry
from quant_fund.models.nutritional_epidemiology import bench_nutritional_epidemiology
from quant_fund.models.sports_nutrition import bench_sports_nutrition

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


def bench_nutritional_biochemistry_family(seed: int = _SEED + 44000) -> dict[str, float]:
    return _finite_blob(bench_nutritional_biochemistry(seed))


def bench_dietary_assessment_family(seed: int = _SEED + 44001) -> dict[str, float]:
    return _finite_blob(bench_dietary_assessment(seed))


def bench_clinical_nutrition_family(seed: int = _SEED + 44002) -> dict[str, float]:
    return _finite_blob(bench_clinical_nutrition(seed))


def bench_sports_nutrition_family(seed: int = _SEED + 44003) -> dict[str, float]:
    return _finite_blob(bench_sports_nutrition(seed))


def bench_nutritional_epidemiology_family(seed: int = _SEED + 44004) -> dict[str, float]:
    return _finite_blob(bench_nutritional_epidemiology(seed))


def bench_metabolic_health_family(seed: int = _SEED + 44005) -> dict[str, float]:
    return _finite_blob(bench_metabolic_health(seed))
