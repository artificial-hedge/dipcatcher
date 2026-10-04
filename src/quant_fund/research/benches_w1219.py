"""Wave-1219 hem-onc canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.hematologic_malignancies import bench_hematologic_malignancies
from quant_fund.models.hematology_studies import bench_hematology_studies
from quant_fund.models.oncology_studies import bench_oncology_studies
from quant_fund.models.radiation_oncology import bench_radiation_oncology
from quant_fund.models.solid_tumor_oncology import bench_solid_tumor_oncology
from quant_fund.models.transfusion_medicine import bench_transfusion_medicine

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


def bench_hematology_studies_family(seed: int = _SEED + 60700) -> dict[str, float]:
    return _finite_blob(bench_hematology_studies(seed))


def bench_oncology_studies_family(seed: int = _SEED + 60701) -> dict[str, float]:
    return _finite_blob(bench_oncology_studies(seed))


def bench_hematologic_malignancies_family(seed: int = _SEED + 60702) -> dict[str, float]:
    return _finite_blob(bench_hematologic_malignancies(seed))


def bench_solid_tumor_oncology_family(seed: int = _SEED + 60703) -> dict[str, float]:
    return _finite_blob(bench_solid_tumor_oncology(seed))


def bench_transfusion_medicine_family(seed: int = _SEED + 60704) -> dict[str, float]:
    return _finite_blob(bench_transfusion_medicine(seed))


def bench_radiation_oncology_family(seed: int = _SEED + 60705) -> dict[str, float]:
    return _finite_blob(bench_radiation_oncology(seed))
