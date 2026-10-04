"""Wave-1102 geology-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.geophysics_applied import bench_geophysics_applied
from quant_fund.models.hydrogeology import bench_hydrogeology
from quant_fund.models.mineralogy import bench_mineralogy
from quant_fund.models.sedimentology import bench_sedimentology
from quant_fund.models.tectonics import bench_tectonics
from quant_fund.models.volcanology import bench_volcanology

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


def bench_mineralogy_family(seed: int = _SEED + 49000) -> dict[str, float]:
    return _finite_blob(bench_mineralogy(seed))


def bench_volcanology_family(seed: int = _SEED + 49001) -> dict[str, float]:
    return _finite_blob(bench_volcanology(seed))


def bench_sedimentology_family(seed: int = _SEED + 49002) -> dict[str, float]:
    return _finite_blob(bench_sedimentology(seed))


def bench_tectonics_family(seed: int = _SEED + 49003) -> dict[str, float]:
    return _finite_blob(bench_tectonics(seed))


def bench_hydrogeology_family(seed: int = _SEED + 49004) -> dict[str, float]:
    return _finite_blob(bench_hydrogeology(seed))


def bench_geophysics_applied_family(seed: int = _SEED + 49005) -> dict[str, float]:
    return _finite_blob(bench_geophysics_applied(seed))
