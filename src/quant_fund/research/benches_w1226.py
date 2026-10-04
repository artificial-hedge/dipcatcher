"""Wave-1226 pathology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.anatomical_pathology import bench_anatomical_pathology
from quant_fund.models.clinical_pathology import bench_clinical_pathology
from quant_fund.models.cytopathology import bench_cytopathology
from quant_fund.models.histopathology_studies import bench_histopathology_studies
from quant_fund.models.molecular_pathology import bench_molecular_pathology
from quant_fund.models.pathology_studies import bench_pathology_studies

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


def bench_pathology_studies_family(seed: int = _SEED + 61400) -> dict[str, float]:
    return _finite_blob(bench_pathology_studies(seed))


def bench_anatomical_pathology_family(seed: int = _SEED + 61401) -> dict[str, float]:
    return _finite_blob(bench_anatomical_pathology(seed))


def bench_clinical_pathology_family(seed: int = _SEED + 61402) -> dict[str, float]:
    return _finite_blob(bench_clinical_pathology(seed))


def bench_histopathology_studies_family(seed: int = _SEED + 61403) -> dict[str, float]:
    return _finite_blob(bench_histopathology_studies(seed))


def bench_cytopathology_family(seed: int = _SEED + 61404) -> dict[str, float]:
    return _finite_blob(bench_cytopathology(seed))


def bench_molecular_pathology_family(seed: int = _SEED + 61405) -> dict[str, float]:
    return _finite_blob(bench_molecular_pathology(seed))
