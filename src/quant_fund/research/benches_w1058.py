"""Wave-1058 philosophy canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.aesthetics import bench_aesthetics
from quant_fund.models.epistemology import bench_epistemology
from quant_fund.models.ethics_philosophy import bench_ethics_philosophy
from quant_fund.models.logic_philosophy import bench_logic_philosophy
from quant_fund.models.metaphysics import bench_metaphysics
from quant_fund.models.philosophy_of_science import bench_philosophy_of_science

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


def bench_metaphysics_family(seed: int = _SEED + 44600) -> dict[str, float]:
    return _finite_blob(bench_metaphysics(seed))


def bench_epistemology_family(seed: int = _SEED + 44601) -> dict[str, float]:
    return _finite_blob(bench_epistemology(seed))


def bench_ethics_philosophy_family(seed: int = _SEED + 44602) -> dict[str, float]:
    return _finite_blob(bench_ethics_philosophy(seed))


def bench_logic_philosophy_family(seed: int = _SEED + 44603) -> dict[str, float]:
    return _finite_blob(bench_logic_philosophy(seed))


def bench_philosophy_of_science_family(seed: int = _SEED + 44604) -> dict[str, float]:
    return _finite_blob(bench_philosophy_of_science(seed))


def bench_aesthetics_family(seed: int = _SEED + 44605) -> dict[str, float]:
    return _finite_blob(bench_aesthetics(seed))
