"""Wave-1217 endocrinology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.adrenal_medicine import bench_adrenal_medicine
from quant_fund.models.bone_metabolism import bench_bone_metabolism
from quant_fund.models.diabetes_medicine import bench_diabetes_medicine
from quant_fund.models.endocrinology_studies import bench_endocrinology_studies
from quant_fund.models.metabolic_medicine import bench_metabolic_medicine
from quant_fund.models.thyroid_medicine import bench_thyroid_medicine

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


def bench_endocrinology_studies_family(seed: int = _SEED + 60500) -> dict[str, float]:
    return _finite_blob(bench_endocrinology_studies(seed))


def bench_diabetes_medicine_family(seed: int = _SEED + 60501) -> dict[str, float]:
    return _finite_blob(bench_diabetes_medicine(seed))


def bench_thyroid_medicine_family(seed: int = _SEED + 60502) -> dict[str, float]:
    return _finite_blob(bench_thyroid_medicine(seed))


def bench_metabolic_medicine_family(seed: int = _SEED + 60503) -> dict[str, float]:
    return _finite_blob(bench_metabolic_medicine(seed))


def bench_bone_metabolism_family(seed: int = _SEED + 60504) -> dict[str, float]:
    return _finite_blob(bench_bone_metabolism(seed))


def bench_adrenal_medicine_family(seed: int = _SEED + 60505) -> dict[str, float]:
    return _finite_blob(bench_adrenal_medicine(seed))
