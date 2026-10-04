"""Wave-1231 transplant/immunology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.allergy_studies import bench_allergy_studies
from quant_fund.models.autoimmunity_studies import bench_autoimmunity_studies
from quant_fund.models.hematopoietic_transplant import bench_hematopoietic_transplant
from quant_fund.models.immunodeficiency_studies import bench_immunodeficiency_studies
from quant_fund.models.immunology_medicine import bench_immunology_medicine
from quant_fund.models.transplant_medicine_studies import bench_transplant_medicine_studies

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


def bench_transplant_medicine_studies_family(seed: int = _SEED + 61900) -> dict[str, float]:
    return _finite_blob(bench_transplant_medicine_studies(seed))


def bench_immunology_medicine_family(seed: int = _SEED + 61901) -> dict[str, float]:
    return _finite_blob(bench_immunology_medicine(seed))


def bench_allergy_studies_family(seed: int = _SEED + 61902) -> dict[str, float]:
    return _finite_blob(bench_allergy_studies(seed))


def bench_autoimmunity_studies_family(seed: int = _SEED + 61903) -> dict[str, float]:
    return _finite_blob(bench_autoimmunity_studies(seed))


def bench_hematopoietic_transplant_family(seed: int = _SEED + 61904) -> dict[str, float]:
    return _finite_blob(bench_hematopoietic_transplant(seed))


def bench_immunodeficiency_studies_family(seed: int = _SEED + 61905) -> dict[str, float]:
    return _finite_blob(bench_immunodeficiency_studies(seed))
