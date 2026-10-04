"""Wave-1149 clinical-medicine canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.dermatology_2 import bench_dermatology_2
from quant_fund.models.hematology_2 import bench_hematology_2
from quant_fund.models.hepatology_2 import bench_hepatology_2
from quant_fund.models.nephrology_2 import bench_nephrology_2
from quant_fund.models.pulmonology_2 import bench_pulmonology_2
from quant_fund.models.toxicology_2 import bench_toxicology_2

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


def bench_toxicology_2_family(seed: int = _SEED + 53700) -> dict[str, float]:
    return _finite_blob(bench_toxicology_2(seed))


def bench_dermatology_2_family(seed: int = _SEED + 53701) -> dict[str, float]:
    return _finite_blob(bench_dermatology_2(seed))


def bench_hematology_2_family(seed: int = _SEED + 53702) -> dict[str, float]:
    return _finite_blob(bench_hematology_2(seed))


def bench_pulmonology_2_family(seed: int = _SEED + 53703) -> dict[str, float]:
    return _finite_blob(bench_pulmonology_2(seed))


def bench_nephrology_2_family(seed: int = _SEED + 53704) -> dict[str, float]:
    return _finite_blob(bench_nephrology_2(seed))


def bench_hepatology_2_family(seed: int = _SEED + 53705) -> dict[str, float]:
    return _finite_blob(bench_hepatology_2(seed))
