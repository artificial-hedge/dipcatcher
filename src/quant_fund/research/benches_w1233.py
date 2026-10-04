"""Wave-1233 clinical-genetics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.dysmorphology_studies import bench_dysmorphology_studies
from quant_fund.models.genetic_diagnostics import bench_genetic_diagnostics
from quant_fund.models.lysosomal_medicine import bench_lysosomal_medicine
from quant_fund.models.medical_genetics_studies import bench_medical_genetics_studies
from quant_fund.models.mitochondrial_medicine import bench_mitochondrial_medicine
from quant_fund.models.pharmacogenomics_studies import bench_pharmacogenomics_studies

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


def bench_medical_genetics_studies_family(seed: int = _SEED + 62100) -> dict[str, float]:
    return _finite_blob(bench_medical_genetics_studies(seed))


def bench_genetic_diagnostics_family(seed: int = _SEED + 62101) -> dict[str, float]:
    return _finite_blob(bench_genetic_diagnostics(seed))


def bench_lysosomal_medicine_family(seed: int = _SEED + 62102) -> dict[str, float]:
    return _finite_blob(bench_lysosomal_medicine(seed))


def bench_mitochondrial_medicine_family(seed: int = _SEED + 62103) -> dict[str, float]:
    return _finite_blob(bench_mitochondrial_medicine(seed))


def bench_dysmorphology_studies_family(seed: int = _SEED + 62104) -> dict[str, float]:
    return _finite_blob(bench_dysmorphology_studies(seed))


def bench_pharmacogenomics_studies_family(seed: int = _SEED + 62105) -> dict[str, float]:
    return _finite_blob(bench_pharmacogenomics_studies(seed))
