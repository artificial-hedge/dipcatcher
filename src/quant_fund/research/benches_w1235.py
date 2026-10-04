"""Wave-1235 rheumatology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.connective_tissue_studies import bench_connective_tissue_studies
from quant_fund.models.inflammatory_arthritis_studies import bench_inflammatory_arthritis_studies
from quant_fund.models.myositis_studies import bench_myositis_studies
from quant_fund.models.osteoarthritis_studies import bench_osteoarthritis_studies
from quant_fund.models.rheumatology_medicine import bench_rheumatology_medicine
from quant_fund.models.spondyloarthritis_studies import bench_spondyloarthritis_studies

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


def bench_rheumatology_medicine_family(seed: int = _SEED + 62300) -> dict[str, float]:
    return _finite_blob(bench_rheumatology_medicine(seed))


def bench_spondyloarthritis_studies_family(seed: int = _SEED + 62301) -> dict[str, float]:
    return _finite_blob(bench_spondyloarthritis_studies(seed))


def bench_inflammatory_arthritis_studies_family(seed: int = _SEED + 62302) -> dict[str, float]:
    return _finite_blob(bench_inflammatory_arthritis_studies(seed))


def bench_connective_tissue_studies_family(seed: int = _SEED + 62303) -> dict[str, float]:
    return _finite_blob(bench_connective_tissue_studies(seed))


def bench_osteoarthritis_studies_family(seed: int = _SEED + 62304) -> dict[str, float]:
    return _finite_blob(bench_osteoarthritis_studies(seed))


def bench_myositis_studies_family(seed: int = _SEED + 62305) -> dict[str, float]:
    return _finite_blob(bench_myositis_studies(seed))
