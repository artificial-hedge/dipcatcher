"""Wave-1152 molecular-life-sciences canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.biochemistry_2 import bench_biochemistry_2
from quant_fund.models.cell_biology_2 import bench_cell_biology_2
from quant_fund.models.genetics_2 import bench_genetics_2
from quant_fund.models.molecular_biology_2 import bench_molecular_biology_2
from quant_fund.models.pharmacology_2 import bench_pharmacology_2
from quant_fund.models.toxicology_3 import bench_toxicology_3

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


def bench_biochemistry_2_family(seed: int = _SEED + 54000) -> dict[str, float]:
    return _finite_blob(bench_biochemistry_2(seed))


def bench_molecular_biology_2_family(seed: int = _SEED + 54001) -> dict[str, float]:
    return _finite_blob(bench_molecular_biology_2(seed))


def bench_cell_biology_2_family(seed: int = _SEED + 54002) -> dict[str, float]:
    return _finite_blob(bench_cell_biology_2(seed))


def bench_genetics_2_family(seed: int = _SEED + 54003) -> dict[str, float]:
    return _finite_blob(bench_genetics_2(seed))


def bench_pharmacology_2_family(seed: int = _SEED + 54004) -> dict[str, float]:
    return _finite_blob(bench_pharmacology_2(seed))


def bench_toxicology_3_family(seed: int = _SEED + 54005) -> dict[str, float]:
    return _finite_blob(bench_toxicology_3(seed))
