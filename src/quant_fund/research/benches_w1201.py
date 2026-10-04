"""Wave-1201 health-informatics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.biomedical_informatics import bench_biomedical_informatics
from quant_fund.models.clinical_informatics import bench_clinical_informatics
from quant_fund.models.health_data_science import bench_health_data_science
from quant_fund.models.health_informatics import bench_health_informatics
from quant_fund.models.health_information import bench_health_information
from quant_fund.models.medical_records import bench_medical_records

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


def bench_health_informatics_family(seed: int = _SEED + 58900) -> dict[str, float]:
    return _finite_blob(bench_health_informatics(seed))


def bench_medical_records_family(seed: int = _SEED + 58901) -> dict[str, float]:
    return _finite_blob(bench_medical_records(seed))


def bench_health_information_family(seed: int = _SEED + 58902) -> dict[str, float]:
    return _finite_blob(bench_health_information(seed))


def bench_biomedical_informatics_family(seed: int = _SEED + 58903) -> dict[str, float]:
    return _finite_blob(bench_biomedical_informatics(seed))


def bench_clinical_informatics_family(seed: int = _SEED + 58904) -> dict[str, float]:
    return _finite_blob(bench_clinical_informatics(seed))


def bench_health_data_science_family(seed: int = _SEED + 58905) -> dict[str, float]:
    return _finite_blob(bench_health_data_science(seed))
