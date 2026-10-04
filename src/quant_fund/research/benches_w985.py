"""Wave-985 Hardy-space/BMO canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.atomic_h1 import bench_atomic_h1
from quant_fund.models.bmo_space import bench_bmo_space
from quant_fund.models.carleson_measure import bench_carleson_measure
from quant_fund.models.fefferman_stein import bench_fefferman_stein
from quant_fund.models.hardy_h1 import bench_hardy_h1
from quant_fund.models.john_nirenberg import bench_john_nirenberg

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


def bench_hardy_h1_family(seed: int = _SEED + 37300) -> dict[str, float]:
    return _finite_blob(bench_hardy_h1(seed))


def bench_bmo_space_family(seed: int = _SEED + 37301) -> dict[str, float]:
    return _finite_blob(bench_bmo_space(seed))


def bench_atomic_h1_family(seed: int = _SEED + 37302) -> dict[str, float]:
    return _finite_blob(bench_atomic_h1(seed))


def bench_carleson_measure_family(seed: int = _SEED + 37303) -> dict[str, float]:
    return _finite_blob(bench_carleson_measure(seed))


def bench_john_nirenberg_family(seed: int = _SEED + 37304) -> dict[str, float]:
    return _finite_blob(bench_john_nirenberg(seed))


def bench_fefferman_stein_family(seed: int = _SEED + 37305) -> dict[str, float]:
    return _finite_blob(bench_fefferman_stein(seed))
