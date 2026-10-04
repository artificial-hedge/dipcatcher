"""Wave-1221 derm-eye-ent canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.audiology_medicine import bench_audiology_medicine
from quant_fund.models.dermatology_studies import bench_dermatology_studies
from quant_fund.models.dermatopathology import bench_dermatopathology
from quant_fund.models.ophthalmology_studies import bench_ophthalmology_studies
from quant_fund.models.optometry_studies import bench_optometry_studies
from quant_fund.models.otolaryngology_studies import bench_otolaryngology_studies

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


def bench_dermatology_studies_family(seed: int = _SEED + 60900) -> dict[str, float]:
    return _finite_blob(bench_dermatology_studies(seed))


def bench_ophthalmology_studies_family(seed: int = _SEED + 60901) -> dict[str, float]:
    return _finite_blob(bench_ophthalmology_studies(seed))


def bench_otolaryngology_studies_family(seed: int = _SEED + 60902) -> dict[str, float]:
    return _finite_blob(bench_otolaryngology_studies(seed))


def bench_audiology_medicine_family(seed: int = _SEED + 60903) -> dict[str, float]:
    return _finite_blob(bench_audiology_medicine(seed))


def bench_optometry_studies_family(seed: int = _SEED + 60904) -> dict[str, float]:
    return _finite_blob(bench_optometry_studies(seed))


def bench_dermatopathology_family(seed: int = _SEED + 60905) -> dict[str, float]:
    return _finite_blob(bench_dermatopathology(seed))
