"""Wave-1214 cardiology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.cardiology_studies import bench_cardiology_studies
from quant_fund.models.cardiovascular_imaging import bench_cardiovascular_imaging
from quant_fund.models.electrophysiology_studies import bench_electrophysiology_studies
from quant_fund.models.heart_failure_medicine import bench_heart_failure_medicine
from quant_fund.models.interventional_cardiology import bench_interventional_cardiology
from quant_fund.models.preventive_cardiology import bench_preventive_cardiology

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


def bench_cardiology_studies_family(seed: int = _SEED + 60200) -> dict[str, float]:
    return _finite_blob(bench_cardiology_studies(seed))


def bench_interventional_cardiology_family(seed: int = _SEED + 60201) -> dict[str, float]:
    return _finite_blob(bench_interventional_cardiology(seed))


def bench_electrophysiology_studies_family(seed: int = _SEED + 60202) -> dict[str, float]:
    return _finite_blob(bench_electrophysiology_studies(seed))


def bench_heart_failure_medicine_family(seed: int = _SEED + 60203) -> dict[str, float]:
    return _finite_blob(bench_heart_failure_medicine(seed))


def bench_preventive_cardiology_family(seed: int = _SEED + 60204) -> dict[str, float]:
    return _finite_blob(bench_preventive_cardiology(seed))


def bench_cardiovascular_imaging_family(seed: int = _SEED + 60205) -> dict[str, float]:
    return _finite_blob(bench_cardiovascular_imaging(seed))
