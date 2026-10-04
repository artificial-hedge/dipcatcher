"""Wave-1195 clinical-support canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.clinical_laboratory import bench_clinical_laboratory
from quant_fund.models.medical_imaging_studies import bench_medical_imaging_studies
from quant_fund.models.mortuary_science import bench_mortuary_science
from quant_fund.models.phlebotomy_studies import bench_phlebotomy_studies
from quant_fund.models.sterile_processing import bench_sterile_processing
from quant_fund.models.surgical_technology import bench_surgical_technology

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


def bench_medical_imaging_studies_family(seed: int = _SEED + 58300) -> dict[str, float]:
    return _finite_blob(bench_medical_imaging_studies(seed))


def bench_clinical_laboratory_family(seed: int = _SEED + 58301) -> dict[str, float]:
    return _finite_blob(bench_clinical_laboratory(seed))


def bench_mortuary_science_family(seed: int = _SEED + 58302) -> dict[str, float]:
    return _finite_blob(bench_mortuary_science(seed))


def bench_phlebotomy_studies_family(seed: int = _SEED + 58303) -> dict[str, float]:
    return _finite_blob(bench_phlebotomy_studies(seed))


def bench_surgical_technology_family(seed: int = _SEED + 58304) -> dict[str, float]:
    return _finite_blob(bench_surgical_technology(seed))


def bench_sterile_processing_family(seed: int = _SEED + 58305) -> dict[str, float]:
    return _finite_blob(bench_sterile_processing(seed))
