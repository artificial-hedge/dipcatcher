"""Wave-1156 mathematical-sciences canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.applied_mathematics import bench_applied_mathematics
from quant_fund.models.bioinformatics_5 import bench_bioinformatics_5
from quant_fund.models.computational_science import bench_computational_science
from quant_fund.models.data_science import bench_data_science
from quant_fund.models.probability_4 import bench_probability_4
from quant_fund.models.statistics_2 import bench_statistics_2

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


def bench_applied_mathematics_family(seed: int = _SEED + 54400) -> dict[str, float]:
    return _finite_blob(bench_applied_mathematics(seed))


def bench_statistics_2_family(seed: int = _SEED + 54401) -> dict[str, float]:
    return _finite_blob(bench_statistics_2(seed))


def bench_probability_4_family(seed: int = _SEED + 54402) -> dict[str, float]:
    return _finite_blob(bench_probability_4(seed))


def bench_computational_science_family(seed: int = _SEED + 54403) -> dict[str, float]:
    return _finite_blob(bench_computational_science(seed))


def bench_data_science_family(seed: int = _SEED + 54404) -> dict[str, float]:
    return _finite_blob(bench_data_science(seed))


def bench_bioinformatics_5_family(seed: int = _SEED + 54405) -> dict[str, float]:
    return _finite_blob(bench_bioinformatics_5(seed))
