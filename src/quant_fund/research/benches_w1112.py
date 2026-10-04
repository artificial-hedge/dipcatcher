"""Wave-1112 biology-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.biophysics import bench_biophysics
from quant_fund.models.comparative_anatomy import bench_comparative_anatomy
from quant_fund.models.developmental_biology import bench_developmental_biology
from quant_fund.models.ethology import bench_ethology
from quant_fund.models.evolutionary_biology import bench_evolutionary_biology
from quant_fund.models.neurobiology import bench_neurobiology

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


def bench_biophysics_family(seed: int = _SEED + 50000) -> dict[str, float]:
    return _finite_blob(bench_biophysics(seed))


def bench_evolutionary_biology_family(seed: int = _SEED + 50001) -> dict[str, float]:
    return _finite_blob(bench_evolutionary_biology(seed))


def bench_developmental_biology_family(seed: int = _SEED + 50002) -> dict[str, float]:
    return _finite_blob(bench_developmental_biology(seed))


def bench_neurobiology_family(seed: int = _SEED + 50003) -> dict[str, float]:
    return _finite_blob(bench_neurobiology(seed))


def bench_ethology_family(seed: int = _SEED + 50004) -> dict[str, float]:
    return _finite_blob(bench_ethology(seed))


def bench_comparative_anatomy_family(seed: int = _SEED + 50005) -> dict[str, float]:
    return _finite_blob(bench_comparative_anatomy(seed))
