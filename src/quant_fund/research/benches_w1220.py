"""Wave-1220 infectious-immune canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.allergy_immunology import bench_allergy_immunology
from quant_fund.models.antimicrobial_stewardship import bench_antimicrobial_stewardship
from quant_fund.models.hiv_medicine import bench_hiv_medicine
from quant_fund.models.immunology_studies import bench_immunology_studies
from quant_fund.models.infectious_disease_medicine import bench_infectious_disease_medicine
from quant_fund.models.rheumatology_studies import bench_rheumatology_studies

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


def bench_infectious_disease_medicine_family(seed: int = _SEED + 60800) -> dict[str, float]:
    return _finite_blob(bench_infectious_disease_medicine(seed))


def bench_hiv_medicine_family(seed: int = _SEED + 60801) -> dict[str, float]:
    return _finite_blob(bench_hiv_medicine(seed))


def bench_antimicrobial_stewardship_family(seed: int = _SEED + 60802) -> dict[str, float]:
    return _finite_blob(bench_antimicrobial_stewardship(seed))


def bench_rheumatology_studies_family(seed: int = _SEED + 60803) -> dict[str, float]:
    return _finite_blob(bench_rheumatology_studies(seed))


def bench_immunology_studies_family(seed: int = _SEED + 60804) -> dict[str, float]:
    return _finite_blob(bench_immunology_studies(seed))


def bench_allergy_immunology_family(seed: int = _SEED + 60805) -> dict[str, float]:
    return _finite_blob(bench_allergy_immunology(seed))
