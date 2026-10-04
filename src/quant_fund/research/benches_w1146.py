"""Wave-1146 organismal-biology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.entomology_2 import bench_entomology_2
from quant_fund.models.limnology import bench_limnology
from quant_fund.models.mycology import bench_mycology
from quant_fund.models.parasitology import bench_parasitology
from quant_fund.models.virology import bench_virology
from quant_fund.models.wildlife_biology import bench_wildlife_biology

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


def bench_virology_family(seed: int = _SEED + 53400) -> dict[str, float]:
    return _finite_blob(bench_virology(seed))


def bench_parasitology_family(seed: int = _SEED + 53401) -> dict[str, float]:
    return _finite_blob(bench_parasitology(seed))


def bench_mycology_family(seed: int = _SEED + 53402) -> dict[str, float]:
    return _finite_blob(bench_mycology(seed))


def bench_entomology_2_family(seed: int = _SEED + 53403) -> dict[str, float]:
    return _finite_blob(bench_entomology_2(seed))


def bench_limnology_family(seed: int = _SEED + 53404) -> dict[str, float]:
    return _finite_blob(bench_limnology(seed))


def bench_wildlife_biology_family(seed: int = _SEED + 53405) -> dict[str, float]:
    return _finite_blob(bench_wildlife_biology(seed))
