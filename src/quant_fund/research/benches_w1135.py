"""Wave-1135 medicine-5 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.ophthalmology import bench_ophthalmology
from quant_fund.models.otolaryngology import bench_otolaryngology
from quant_fund.models.palliative_medicine import bench_palliative_medicine
from quant_fund.models.rehabilitation_medicine import bench_rehabilitation_medicine
from quant_fund.models.sports_medicine import bench_sports_medicine
from quant_fund.models.urology import bench_urology

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


def bench_urology_family(seed: int = _SEED + 52300) -> dict[str, float]:
    return _finite_blob(bench_urology(seed))


def bench_ophthalmology_family(seed: int = _SEED + 52301) -> dict[str, float]:
    return _finite_blob(bench_ophthalmology(seed))


def bench_otolaryngology_family(seed: int = _SEED + 52302) -> dict[str, float]:
    return _finite_blob(bench_otolaryngology(seed))


def bench_palliative_medicine_family(seed: int = _SEED + 52303) -> dict[str, float]:
    return _finite_blob(bench_palliative_medicine(seed))


def bench_sports_medicine_family(seed: int = _SEED + 52304) -> dict[str, float]:
    return _finite_blob(bench_sports_medicine(seed))


def bench_rehabilitation_medicine_family(seed: int = _SEED + 52305) -> dict[str, float]:
    return _finite_blob(bench_rehabilitation_medicine(seed))
