"""Wave-1203 molecular-medicine canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.genomic_medicine import bench_genomic_medicine
from quant_fund.models.laboratory_medicine import bench_laboratory_medicine
from quant_fund.models.molecular_diagnostics import bench_molecular_diagnostics
from quant_fund.models.precision_medicine import bench_precision_medicine
from quant_fund.models.travel_medicine import bench_travel_medicine
from quant_fund.models.tropical_medicine import bench_tropical_medicine

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


def bench_tropical_medicine_family(seed: int = _SEED + 59100) -> dict[str, float]:
    return _finite_blob(bench_tropical_medicine(seed))


def bench_travel_medicine_family(seed: int = _SEED + 59101) -> dict[str, float]:
    return _finite_blob(bench_travel_medicine(seed))


def bench_genomic_medicine_family(seed: int = _SEED + 59102) -> dict[str, float]:
    return _finite_blob(bench_genomic_medicine(seed))


def bench_precision_medicine_family(seed: int = _SEED + 59103) -> dict[str, float]:
    return _finite_blob(bench_precision_medicine(seed))


def bench_molecular_diagnostics_family(seed: int = _SEED + 59104) -> dict[str, float]:
    return _finite_blob(bench_molecular_diagnostics(seed))


def bench_laboratory_medicine_family(seed: int = _SEED + 59105) -> dict[str, float]:
    return _finite_blob(bench_laboratory_medicine(seed))
