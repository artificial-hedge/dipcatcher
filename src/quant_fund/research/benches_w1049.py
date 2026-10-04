"""Wave-1049 dentistry canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.dental_anatomy import bench_dental_anatomy
from quant_fund.models.endodontics import bench_endodontics
from quant_fund.models.oral_pathology import bench_oral_pathology
from quant_fund.models.orthodontics import bench_orthodontics
from quant_fund.models.periodontology import bench_periodontology
from quant_fund.models.prosthodontics import bench_prosthodontics

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


def bench_dental_anatomy_family(seed: int = _SEED + 43700) -> dict[str, float]:
    return _finite_blob(bench_dental_anatomy(seed))


def bench_oral_pathology_family(seed: int = _SEED + 43701) -> dict[str, float]:
    return _finite_blob(bench_oral_pathology(seed))


def bench_periodontology_family(seed: int = _SEED + 43702) -> dict[str, float]:
    return _finite_blob(bench_periodontology(seed))


def bench_endodontics_family(seed: int = _SEED + 43703) -> dict[str, float]:
    return _finite_blob(bench_endodontics(seed))


def bench_orthodontics_family(seed: int = _SEED + 43704) -> dict[str, float]:
    return _finite_blob(bench_orthodontics(seed))


def bench_prosthodontics_family(seed: int = _SEED + 43705) -> dict[str, float]:
    return _finite_blob(bench_prosthodontics(seed))
