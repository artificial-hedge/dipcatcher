"""Wave-1151 chemical-sciences canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.analytical_chemistry_2 import bench_analytical_chemistry_2
from quant_fund.models.chemistry_3 import bench_chemistry_3
from quant_fund.models.electrochemistry_2 import bench_electrochemistry_2
from quant_fund.models.inorganic_chemistry_2 import bench_inorganic_chemistry_2
from quant_fund.models.organic_chemistry_2 import bench_organic_chemistry_2
from quant_fund.models.physical_chemistry_2 import bench_physical_chemistry_2

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


def bench_chemistry_3_family(seed: int = _SEED + 53900) -> dict[str, float]:
    return _finite_blob(bench_chemistry_3(seed))


def bench_organic_chemistry_2_family(seed: int = _SEED + 53901) -> dict[str, float]:
    return _finite_blob(bench_organic_chemistry_2(seed))


def bench_inorganic_chemistry_2_family(seed: int = _SEED + 53902) -> dict[str, float]:
    return _finite_blob(bench_inorganic_chemistry_2(seed))


def bench_physical_chemistry_2_family(seed: int = _SEED + 53903) -> dict[str, float]:
    return _finite_blob(bench_physical_chemistry_2(seed))


def bench_analytical_chemistry_2_family(seed: int = _SEED + 53904) -> dict[str, float]:
    return _finite_blob(bench_analytical_chemistry_2(seed))


def bench_electrochemistry_2_family(seed: int = _SEED + 53905) -> dict[str, float]:
    return _finite_blob(bench_electrochemistry_2(seed))
