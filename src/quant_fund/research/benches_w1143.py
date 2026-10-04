"""Wave-1143 physics-5 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.acoustics_2 import bench_acoustics_2
from quant_fund.models.biophysics_2 import bench_biophysics_2
from quant_fund.models.condensed_matter_3 import bench_condensed_matter_3
from quant_fund.models.nanotechnology import bench_nanotechnology
from quant_fund.models.optics_3 import bench_optics_3
from quant_fund.models.thermodynamics_2 import bench_thermodynamics_2

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


def bench_nanotechnology_family(seed: int = _SEED + 53100) -> dict[str, float]:
    return _finite_blob(bench_nanotechnology(seed))


def bench_biophysics_2_family(seed: int = _SEED + 53101) -> dict[str, float]:
    return _finite_blob(bench_biophysics_2(seed))


def bench_condensed_matter_3_family(seed: int = _SEED + 53102) -> dict[str, float]:
    return _finite_blob(bench_condensed_matter_3(seed))


def bench_optics_3_family(seed: int = _SEED + 53103) -> dict[str, float]:
    return _finite_blob(bench_optics_3(seed))


def bench_acoustics_2_family(seed: int = _SEED + 53104) -> dict[str, float]:
    return _finite_blob(bench_acoustics_2(seed))


def bench_thermodynamics_2_family(seed: int = _SEED + 53105) -> dict[str, float]:
    return _finite_blob(bench_thermodynamics_2(seed))
