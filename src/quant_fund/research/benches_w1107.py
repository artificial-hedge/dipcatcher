"""Wave-1107 medicine-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.endocrinology import bench_endocrinology
from quant_fund.models.gastroenterology import bench_gastroenterology
from quant_fund.models.hematology import bench_hematology
from quant_fund.models.infectious_diseases import bench_infectious_diseases
from quant_fund.models.nephrology import bench_nephrology
from quant_fund.models.pulmonology import bench_pulmonology

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


def bench_gastroenterology_family(seed: int = _SEED + 49500) -> dict[str, float]:
    return _finite_blob(bench_gastroenterology(seed))


def bench_endocrinology_family(seed: int = _SEED + 49501) -> dict[str, float]:
    return _finite_blob(bench_endocrinology(seed))


def bench_hematology_family(seed: int = _SEED + 49502) -> dict[str, float]:
    return _finite_blob(bench_hematology(seed))


def bench_pulmonology_family(seed: int = _SEED + 49503) -> dict[str, float]:
    return _finite_blob(bench_pulmonology(seed))


def bench_nephrology_family(seed: int = _SEED + 49504) -> dict[str, float]:
    return _finite_blob(bench_nephrology(seed))


def bench_infectious_diseases_family(seed: int = _SEED + 49505) -> dict[str, float]:
    return _finite_blob(bench_infectious_diseases(seed))
