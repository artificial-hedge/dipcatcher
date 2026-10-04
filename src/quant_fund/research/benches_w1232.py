"""Wave-1232 geriatrics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.caregiver_medicine import bench_caregiver_medicine
from quant_fund.models.falls_prevention_studies import bench_falls_prevention_studies
from quant_fund.models.frailty_medicine import bench_frailty_medicine
from quant_fund.models.geriatrics_studies import bench_geriatrics_studies
from quant_fund.models.memory_clinic_studies import bench_memory_clinic_studies
from quant_fund.models.polypharmacy_studies import bench_polypharmacy_studies

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


def bench_geriatrics_studies_family(seed: int = _SEED + 62000) -> dict[str, float]:
    return _finite_blob(bench_geriatrics_studies(seed))


def bench_frailty_medicine_family(seed: int = _SEED + 62001) -> dict[str, float]:
    return _finite_blob(bench_frailty_medicine(seed))


def bench_memory_clinic_studies_family(seed: int = _SEED + 62002) -> dict[str, float]:
    return _finite_blob(bench_memory_clinic_studies(seed))


def bench_falls_prevention_studies_family(seed: int = _SEED + 62003) -> dict[str, float]:
    return _finite_blob(bench_falls_prevention_studies(seed))


def bench_polypharmacy_studies_family(seed: int = _SEED + 62004) -> dict[str, float]:
    return _finite_blob(bench_polypharmacy_studies(seed))


def bench_caregiver_medicine_family(seed: int = _SEED + 62005) -> dict[str, float]:
    return _finite_blob(bench_caregiver_medicine(seed))
