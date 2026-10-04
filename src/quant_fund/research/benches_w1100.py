"""Wave-1100 chemistry canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.analytical_chemistry import bench_analytical_chemistry
from quant_fund.models.biochemistry import bench_biochemistry
from quant_fund.models.electrochemistry import bench_electrochemistry
from quant_fund.models.inorganic_chemistry import bench_inorganic_chemistry
from quant_fund.models.organic_chemistry import bench_organic_chemistry
from quant_fund.models.physical_chemistry import bench_physical_chemistry

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


def bench_organic_chemistry_family(seed: int = _SEED + 48800) -> dict[str, float]:
    return _finite_blob(bench_organic_chemistry(seed))


def bench_inorganic_chemistry_family(seed: int = _SEED + 48801) -> dict[str, float]:
    return _finite_blob(bench_inorganic_chemistry(seed))


def bench_physical_chemistry_family(seed: int = _SEED + 48802) -> dict[str, float]:
    return _finite_blob(bench_physical_chemistry(seed))


def bench_analytical_chemistry_family(seed: int = _SEED + 48803) -> dict[str, float]:
    return _finite_blob(bench_analytical_chemistry(seed))


def bench_biochemistry_family(seed: int = _SEED + 48804) -> dict[str, float]:
    return _finite_blob(bench_biochemistry(seed))


def bench_electrochemistry_family(seed: int = _SEED + 48805) -> dict[str, float]:
    return _finite_blob(bench_electrochemistry(seed))
