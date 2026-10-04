"""Wave-1101 biology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.botany import bench_botany
from quant_fund.models.cell_biology import bench_cell_biology
from quant_fund.models.genetics import bench_genetics
from quant_fund.models.microbiology import bench_microbiology
from quant_fund.models.molecular_biology import bench_molecular_biology
from quant_fund.models.zoology import bench_zoology

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


def bench_molecular_biology_family(seed: int = _SEED + 48900) -> dict[str, float]:
    return _finite_blob(bench_molecular_biology(seed))


def bench_cell_biology_family(seed: int = _SEED + 48901) -> dict[str, float]:
    return _finite_blob(bench_cell_biology(seed))


def bench_genetics_family(seed: int = _SEED + 48902) -> dict[str, float]:
    return _finite_blob(bench_genetics(seed))


def bench_microbiology_family(seed: int = _SEED + 48903) -> dict[str, float]:
    return _finite_blob(bench_microbiology(seed))


def bench_zoology_family(seed: int = _SEED + 48904) -> dict[str, float]:
    return _finite_blob(bench_zoology(seed))


def bench_botany_family(seed: int = _SEED + 48905) -> dict[str, float]:
    return _finite_blob(bench_botany(seed))
