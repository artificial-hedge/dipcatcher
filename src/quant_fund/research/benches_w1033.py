"""Wave-1033 biomedical-engineering canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bioinstrumentation import bench_bioinstrumentation
from quant_fund.models.biomechanics import bench_biomechanics
from quant_fund.models.biomedical_imaging2 import bench_biomedical_imaging2
from quant_fund.models.medical_devices import bench_medical_devices
from quant_fund.models.physiological_modeling import bench_physiological_modeling
from quant_fund.models.tissue_engineering import bench_tissue_engineering

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


def bench_biomechanics_family(seed: int = _SEED + 42100) -> dict[str, float]:
    return _finite_blob(bench_biomechanics(seed))


def bench_medical_devices_family(seed: int = _SEED + 42101) -> dict[str, float]:
    return _finite_blob(bench_medical_devices(seed))


def bench_tissue_engineering_family(seed: int = _SEED + 42102) -> dict[str, float]:
    return _finite_blob(bench_tissue_engineering(seed))


def bench_bioinstrumentation_family(seed: int = _SEED + 42103) -> dict[str, float]:
    return _finite_blob(bench_bioinstrumentation(seed))


def bench_physiological_modeling_family(seed: int = _SEED + 42104) -> dict[str, float]:
    return _finite_blob(bench_physiological_modeling(seed))


def bench_biomedical_imaging2_family(seed: int = _SEED + 42105) -> dict[str, float]:
    return _finite_blob(bench_biomedical_imaging2(seed))
