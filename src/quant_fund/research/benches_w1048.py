"""Wave-1048 veterinary-medicine canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.animal_surgery import bench_animal_surgery
from quant_fund.models.equine_medicine import bench_equine_medicine
from quant_fund.models.veterinary_anatomy import bench_veterinary_anatomy
from quant_fund.models.veterinary_epidemiology import bench_veterinary_epidemiology
from quant_fund.models.veterinary_pathology import bench_veterinary_pathology
from quant_fund.models.veterinary_pharmacology import bench_veterinary_pharmacology

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


def bench_veterinary_anatomy_family(seed: int = _SEED + 43600) -> dict[str, float]:
    return _finite_blob(bench_veterinary_anatomy(seed))


def bench_veterinary_pathology_family(seed: int = _SEED + 43601) -> dict[str, float]:
    return _finite_blob(bench_veterinary_pathology(seed))


def bench_veterinary_pharmacology_family(seed: int = _SEED + 43602) -> dict[str, float]:
    return _finite_blob(bench_veterinary_pharmacology(seed))


def bench_animal_surgery_family(seed: int = _SEED + 43603) -> dict[str, float]:
    return _finite_blob(bench_animal_surgery(seed))


def bench_veterinary_epidemiology_family(seed: int = _SEED + 43604) -> dict[str, float]:
    return _finite_blob(bench_veterinary_epidemiology(seed))


def bench_equine_medicine_family(seed: int = _SEED + 43605) -> dict[str, float]:
    return _finite_blob(bench_equine_medicine(seed))
