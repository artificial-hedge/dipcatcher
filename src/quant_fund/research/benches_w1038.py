"""Wave-1038 medicine canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.cardiology import bench_cardiology
from quant_fund.models.human_physiology import bench_human_physiology
from quant_fund.models.immunology import bench_immunology
from quant_fund.models.neuroscience_med import bench_neuroscience_med
from quant_fund.models.pathology import bench_pathology
from quant_fund.models.pharmacokinetics import bench_pharmacokinetics

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


def bench_human_physiology_family(seed: int = _SEED + 42600) -> dict[str, float]:
    return _finite_blob(bench_human_physiology(seed))


def bench_pharmacokinetics_family(seed: int = _SEED + 42601) -> dict[str, float]:
    return _finite_blob(bench_pharmacokinetics(seed))


def bench_immunology_family(seed: int = _SEED + 42602) -> dict[str, float]:
    return _finite_blob(bench_immunology(seed))


def bench_pathology_family(seed: int = _SEED + 42603) -> dict[str, float]:
    return _finite_blob(bench_pathology(seed))


def bench_neuroscience_med_family(seed: int = _SEED + 42604) -> dict[str, float]:
    return _finite_blob(bench_neuroscience_med(seed))


def bench_cardiology_family(seed: int = _SEED + 42605) -> dict[str, float]:
    return _finite_blob(bench_cardiology(seed))
