"""Wave-1161 health-sciences canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.dentistry_3 import bench_dentistry_3
from quant_fund.models.medicine_7 import bench_medicine_7
from quant_fund.models.nursing_2 import bench_nursing_2
from quant_fund.models.pharmacy_2 import bench_pharmacy_2
from quant_fund.models.public_health_2 import bench_public_health_2
from quant_fund.models.veterinary_medicine_2 import bench_veterinary_medicine_2

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


def bench_medicine_7_family(seed: int = _SEED + 54900) -> dict[str, float]:
    return _finite_blob(bench_medicine_7(seed))


def bench_dentistry_3_family(seed: int = _SEED + 54901) -> dict[str, float]:
    return _finite_blob(bench_dentistry_3(seed))


def bench_nursing_2_family(seed: int = _SEED + 54902) -> dict[str, float]:
    return _finite_blob(bench_nursing_2(seed))


def bench_public_health_2_family(seed: int = _SEED + 54903) -> dict[str, float]:
    return _finite_blob(bench_public_health_2(seed))


def bench_veterinary_medicine_2_family(seed: int = _SEED + 54904) -> dict[str, float]:
    return _finite_blob(bench_veterinary_medicine_2(seed))


def bench_pharmacy_2_family(seed: int = _SEED + 54905) -> dict[str, float]:
    return _finite_blob(bench_pharmacy_2(seed))
