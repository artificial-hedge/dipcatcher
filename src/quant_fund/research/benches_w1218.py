"""Wave-1218 nephrology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.acid_base_medicine import bench_acid_base_medicine
from quant_fund.models.dialysis_medicine import bench_dialysis_medicine
from quant_fund.models.hypertension_medicine import bench_hypertension_medicine
from quant_fund.models.nephrology_studies import bench_nephrology_studies
from quant_fund.models.renal_transplant import bench_renal_transplant
from quant_fund.models.urology_studies import bench_urology_studies

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


def bench_nephrology_studies_family(seed: int = _SEED + 60600) -> dict[str, float]:
    return _finite_blob(bench_nephrology_studies(seed))


def bench_dialysis_medicine_family(seed: int = _SEED + 60601) -> dict[str, float]:
    return _finite_blob(bench_dialysis_medicine(seed))


def bench_renal_transplant_family(seed: int = _SEED + 60602) -> dict[str, float]:
    return _finite_blob(bench_renal_transplant(seed))


def bench_acid_base_medicine_family(seed: int = _SEED + 60603) -> dict[str, float]:
    return _finite_blob(bench_acid_base_medicine(seed))


def bench_hypertension_medicine_family(seed: int = _SEED + 60604) -> dict[str, float]:
    return _finite_blob(bench_hypertension_medicine(seed))


def bench_urology_studies_family(seed: int = _SEED + 60605) -> dict[str, float]:
    return _finite_blob(bench_urology_studies(seed))
