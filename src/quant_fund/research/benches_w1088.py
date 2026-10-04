"""Wave-1088 philosophy-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.analytic_philosophy import bench_analytic_philosophy
from quant_fund.models.ancient_philosophy import bench_ancient_philosophy
from quant_fund.models.continental_philosophy import bench_continental_philosophy
from quant_fund.models.existentialism import bench_existentialism
from quant_fund.models.medieval_philosophy import bench_medieval_philosophy
from quant_fund.models.pragmatism import bench_pragmatism

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


def bench_ancient_philosophy_family(seed: int = _SEED + 47600) -> dict[str, float]:
    return _finite_blob(bench_ancient_philosophy(seed))


def bench_medieval_philosophy_family(seed: int = _SEED + 47601) -> dict[str, float]:
    return _finite_blob(bench_medieval_philosophy(seed))


def bench_continental_philosophy_family(seed: int = _SEED + 47602) -> dict[str, float]:
    return _finite_blob(bench_continental_philosophy(seed))


def bench_analytic_philosophy_family(seed: int = _SEED + 47603) -> dict[str, float]:
    return _finite_blob(bench_analytic_philosophy(seed))


def bench_pragmatism_family(seed: int = _SEED + 47604) -> dict[str, float]:
    return _finite_blob(bench_pragmatism(seed))


def bench_existentialism_family(seed: int = _SEED + 47605) -> dict[str, float]:
    return _finite_blob(bench_existentialism(seed))
