"""Wave-1216 internal-medicine canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.critical_care_medicine import bench_critical_care_medicine
from quant_fund.models.gastroenterology_studies import bench_gastroenterology_studies
from quant_fund.models.hepatology_studies import bench_hepatology_studies
from quant_fund.models.hospital_medicine import bench_hospital_medicine
from quant_fund.models.internal_medicine import bench_internal_medicine
from quant_fund.models.pulmonary_medicine import bench_pulmonary_medicine

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


def bench_internal_medicine_family(seed: int = _SEED + 60400) -> dict[str, float]:
    return _finite_blob(bench_internal_medicine(seed))


def bench_hospital_medicine_family(seed: int = _SEED + 60401) -> dict[str, float]:
    return _finite_blob(bench_hospital_medicine(seed))


def bench_critical_care_medicine_family(seed: int = _SEED + 60402) -> dict[str, float]:
    return _finite_blob(bench_critical_care_medicine(seed))


def bench_pulmonary_medicine_family(seed: int = _SEED + 60403) -> dict[str, float]:
    return _finite_blob(bench_pulmonary_medicine(seed))


def bench_gastroenterology_studies_family(seed: int = _SEED + 60404) -> dict[str, float]:
    return _finite_blob(bench_gastroenterology_studies(seed))


def bench_hepatology_studies_family(seed: int = _SEED + 60405) -> dict[str, float]:
    return _finite_blob(bench_hepatology_studies(seed))
