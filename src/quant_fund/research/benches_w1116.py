"""Wave-1116 psychology-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.comparative_psychology import bench_comparative_psychology
from quant_fund.models.environmental_psychology import bench_environmental_psychology
from quant_fund.models.evolutionary_psychology import bench_evolutionary_psychology
from quant_fund.models.experimental_psychology import bench_experimental_psychology
from quant_fund.models.psychopathology import bench_psychopathology
from quant_fund.models.sport_psychology import bench_sport_psychology

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


def bench_experimental_psychology_family(seed: int = _SEED + 50400) -> dict[str, float]:
    return _finite_blob(bench_experimental_psychology(seed))


def bench_comparative_psychology_family(seed: int = _SEED + 50401) -> dict[str, float]:
    return _finite_blob(bench_comparative_psychology(seed))


def bench_evolutionary_psychology_family(seed: int = _SEED + 50402) -> dict[str, float]:
    return _finite_blob(bench_evolutionary_psychology(seed))


def bench_psychopathology_family(seed: int = _SEED + 50403) -> dict[str, float]:
    return _finite_blob(bench_psychopathology(seed))


def bench_environmental_psychology_family(seed: int = _SEED + 50404) -> dict[str, float]:
    return _finite_blob(bench_environmental_psychology(seed))


def bench_sport_psychology_family(seed: int = _SEED + 50405) -> dict[str, float]:
    return _finite_blob(bench_sport_psychology(seed))
