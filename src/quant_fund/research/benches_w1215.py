"""Wave-1215 cardio-surgery canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.adult_congenital import bench_adult_congenital
from quant_fund.models.cardiac_surgery import bench_cardiac_surgery
from quant_fund.models.structural_heart import bench_structural_heart
from quant_fund.models.thoracic_surgery import bench_thoracic_surgery
from quant_fund.models.transplant_cardiology import bench_transplant_cardiology
from quant_fund.models.vascular_surgery import bench_vascular_surgery

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


def bench_vascular_surgery_family(seed: int = _SEED + 60300) -> dict[str, float]:
    return _finite_blob(bench_vascular_surgery(seed))


def bench_cardiac_surgery_family(seed: int = _SEED + 60301) -> dict[str, float]:
    return _finite_blob(bench_cardiac_surgery(seed))


def bench_thoracic_surgery_family(seed: int = _SEED + 60302) -> dict[str, float]:
    return _finite_blob(bench_thoracic_surgery(seed))


def bench_transplant_cardiology_family(seed: int = _SEED + 60303) -> dict[str, float]:
    return _finite_blob(bench_transplant_cardiology(seed))


def bench_structural_heart_family(seed: int = _SEED + 60304) -> dict[str, float]:
    return _finite_blob(bench_structural_heart(seed))


def bench_adult_congenital_family(seed: int = _SEED + 60305) -> dict[str, float]:
    return _finite_blob(bench_adult_congenital(seed))
