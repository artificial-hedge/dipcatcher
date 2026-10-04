"""Wave-1113 medicine-4 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.anesthesiology import bench_anesthesiology
from quant_fund.models.emergency_medicine import bench_emergency_medicine
from quant_fund.models.family_medicine import bench_family_medicine
from quant_fund.models.obstetrics_gynecology import bench_obstetrics_gynecology
from quant_fund.models.pediatrics import bench_pediatrics
from quant_fund.models.surgery import bench_surgery

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


def bench_surgery_family(seed: int = _SEED + 50100) -> dict[str, float]:
    return _finite_blob(bench_surgery(seed))


def bench_anesthesiology_family(seed: int = _SEED + 50101) -> dict[str, float]:
    return _finite_blob(bench_anesthesiology(seed))


def bench_obstetrics_gynecology_family(seed: int = _SEED + 50102) -> dict[str, float]:
    return _finite_blob(bench_obstetrics_gynecology(seed))


def bench_pediatrics_family(seed: int = _SEED + 50103) -> dict[str, float]:
    return _finite_blob(bench_pediatrics(seed))


def bench_emergency_medicine_family(seed: int = _SEED + 50104) -> dict[str, float]:
    return _finite_blob(bench_emergency_medicine(seed))


def bench_family_medicine_family(seed: int = _SEED + 50105) -> dict[str, float]:
    return _finite_blob(bench_family_medicine(seed))
