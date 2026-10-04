"""Wave-1076 archaeology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.archaeometry import bench_archaeometry
from quant_fund.models.bioarchaeology import bench_bioarchaeology
from quant_fund.models.experimental_archaeology import bench_experimental_archaeology
from quant_fund.models.field_archaeology import bench_field_archaeology
from quant_fund.models.landscape_archaeology import bench_landscape_archaeology
from quant_fund.models.underwater_archaeology import bench_underwater_archaeology

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


def bench_field_archaeology_family(seed: int = _SEED + 46400) -> dict[str, float]:
    return _finite_blob(bench_field_archaeology(seed))


def bench_archaeometry_family(seed: int = _SEED + 46401) -> dict[str, float]:
    return _finite_blob(bench_archaeometry(seed))


def bench_bioarchaeology_family(seed: int = _SEED + 46402) -> dict[str, float]:
    return _finite_blob(bench_bioarchaeology(seed))


def bench_underwater_archaeology_family(seed: int = _SEED + 46403) -> dict[str, float]:
    return _finite_blob(bench_underwater_archaeology(seed))


def bench_landscape_archaeology_family(seed: int = _SEED + 46404) -> dict[str, float]:
    return _finite_blob(bench_landscape_archaeology(seed))


def bench_experimental_archaeology_family(seed: int = _SEED + 46405) -> dict[str, float]:
    return _finite_blob(bench_experimental_archaeology(seed))
