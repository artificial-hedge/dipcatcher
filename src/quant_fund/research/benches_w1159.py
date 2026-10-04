"""Wave-1159 engineering canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.aerospace_engineering_2 import bench_aerospace_engineering_2
from quant_fund.models.biomedical_engineering_2 import bench_biomedical_engineering_2
from quant_fund.models.chemical_engineering_2 import bench_chemical_engineering_2
from quant_fund.models.civil_engineering_2 import bench_civil_engineering_2
from quant_fund.models.electrical_engineering_2 import bench_electrical_engineering_2
from quant_fund.models.mechanical_engineering_2 import bench_mechanical_engineering_2

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


def bench_biomedical_engineering_2_family(seed: int = _SEED + 54700) -> dict[str, float]:
    return _finite_blob(bench_biomedical_engineering_2(seed))


def bench_chemical_engineering_2_family(seed: int = _SEED + 54701) -> dict[str, float]:
    return _finite_blob(bench_chemical_engineering_2(seed))


def bench_mechanical_engineering_2_family(seed: int = _SEED + 54702) -> dict[str, float]:
    return _finite_blob(bench_mechanical_engineering_2(seed))


def bench_civil_engineering_2_family(seed: int = _SEED + 54703) -> dict[str, float]:
    return _finite_blob(bench_civil_engineering_2(seed))


def bench_electrical_engineering_2_family(seed: int = _SEED + 54704) -> dict[str, float]:
    return _finite_blob(bench_electrical_engineering_2(seed))


def bench_aerospace_engineering_2_family(seed: int = _SEED + 54705) -> dict[str, float]:
    return _finite_blob(bench_aerospace_engineering_2(seed))
