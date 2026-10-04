"""Wave-1198 procedural-medicine canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.electrodiagnostic_studies import bench_electrodiagnostic_studies
from quant_fund.models.hyperbaric_medicine import bench_hyperbaric_medicine
from quant_fund.models.infusion_therapy import bench_infusion_therapy
from quant_fund.models.pain_management import bench_pain_management
from quant_fund.models.sleep_medicine import bench_sleep_medicine
from quant_fund.models.wound_care import bench_wound_care

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


def bench_sleep_medicine_family(seed: int = _SEED + 58600) -> dict[str, float]:
    return _finite_blob(bench_sleep_medicine(seed))


def bench_pain_management_family(seed: int = _SEED + 58601) -> dict[str, float]:
    return _finite_blob(bench_pain_management(seed))


def bench_wound_care_family(seed: int = _SEED + 58602) -> dict[str, float]:
    return _finite_blob(bench_wound_care(seed))


def bench_infusion_therapy_family(seed: int = _SEED + 58603) -> dict[str, float]:
    return _finite_blob(bench_infusion_therapy(seed))


def bench_hyperbaric_medicine_family(seed: int = _SEED + 58604) -> dict[str, float]:
    return _finite_blob(bench_hyperbaric_medicine(seed))


def bench_electrodiagnostic_studies_family(seed: int = _SEED + 58605) -> dict[str, float]:
    return _finite_blob(bench_electrodiagnostic_studies(seed))
