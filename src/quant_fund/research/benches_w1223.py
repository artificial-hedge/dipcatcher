"""Wave-1223 anesthesia canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.airway_management import bench_airway_management
from quant_fund.models.anesthesiology_studies import bench_anesthesiology_studies
from quant_fund.models.pain_medicine_studies import bench_pain_medicine_studies
from quant_fund.models.perioperative_medicine import bench_perioperative_medicine
from quant_fund.models.regional_anesthesia import bench_regional_anesthesia
from quant_fund.models.sedation_medicine import bench_sedation_medicine

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


def bench_anesthesiology_studies_family(seed: int = _SEED + 61100) -> dict[str, float]:
    return _finite_blob(bench_anesthesiology_studies(seed))


def bench_perioperative_medicine_family(seed: int = _SEED + 61101) -> dict[str, float]:
    return _finite_blob(bench_perioperative_medicine(seed))


def bench_pain_medicine_studies_family(seed: int = _SEED + 61102) -> dict[str, float]:
    return _finite_blob(bench_pain_medicine_studies(seed))


def bench_regional_anesthesia_family(seed: int = _SEED + 61103) -> dict[str, float]:
    return _finite_blob(bench_regional_anesthesia(seed))


def bench_sedation_medicine_family(seed: int = _SEED + 61104) -> dict[str, float]:
    return _finite_blob(bench_sedation_medicine(seed))


def bench_airway_management_family(seed: int = _SEED + 61105) -> dict[str, float]:
    return _finite_blob(bench_airway_management(seed))
