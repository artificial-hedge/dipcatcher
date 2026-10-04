"""Wave-1110 philosophy-4 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.phenomenology_2 import bench_phenomenology_2
from quant_fund.models.philosophy_of_biology import bench_philosophy_of_biology
from quant_fund.models.philosophy_of_history import bench_philosophy_of_history
from quant_fund.models.philosophy_of_mathematics import bench_philosophy_of_mathematics
from quant_fund.models.philosophy_of_religion import bench_philosophy_of_religion
from quant_fund.models.process_philosophy import bench_process_philosophy

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


def bench_philosophy_of_biology_family(seed: int = _SEED + 49800) -> dict[str, float]:
    return _finite_blob(bench_philosophy_of_biology(seed))


def bench_philosophy_of_mathematics_family(seed: int = _SEED + 49801) -> dict[str, float]:
    return _finite_blob(bench_philosophy_of_mathematics(seed))


def bench_philosophy_of_religion_family(seed: int = _SEED + 49802) -> dict[str, float]:
    return _finite_blob(bench_philosophy_of_religion(seed))


def bench_phenomenology_2_family(seed: int = _SEED + 49803) -> dict[str, float]:
    return _finite_blob(bench_phenomenology_2(seed))


def bench_philosophy_of_history_family(seed: int = _SEED + 49804) -> dict[str, float]:
    return _finite_blob(bench_philosophy_of_history(seed))


def bench_process_philosophy_family(seed: int = _SEED + 49805) -> dict[str, float]:
    return _finite_blob(bench_process_philosophy(seed))
