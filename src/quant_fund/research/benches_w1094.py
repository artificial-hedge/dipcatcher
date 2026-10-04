"""Wave-1094 medicine-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.dermatology import bench_dermatology
from quant_fund.models.neurology import bench_neurology
from quant_fund.models.oncology import bench_oncology
from quant_fund.models.orthopedics import bench_orthopedics
from quant_fund.models.psychiatry import bench_psychiatry
from quant_fund.models.radiology import bench_radiology

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


def bench_oncology_family(seed: int = _SEED + 48200) -> dict[str, float]:
    return _finite_blob(bench_oncology(seed))


def bench_neurology_family(seed: int = _SEED + 48201) -> dict[str, float]:
    return _finite_blob(bench_neurology(seed))


def bench_dermatology_family(seed: int = _SEED + 48202) -> dict[str, float]:
    return _finite_blob(bench_dermatology(seed))


def bench_orthopedics_family(seed: int = _SEED + 48203) -> dict[str, float]:
    return _finite_blob(bench_orthopedics(seed))


def bench_psychiatry_family(seed: int = _SEED + 48204) -> dict[str, float]:
    return _finite_blob(bench_psychiatry(seed))


def bench_radiology_family(seed: int = _SEED + 48205) -> dict[str, float]:
    return _finite_blob(bench_radiology(seed))
