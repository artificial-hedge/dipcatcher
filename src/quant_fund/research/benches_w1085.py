"""Wave-1085 literary-periods canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.medieval_literature import bench_medieval_literature
from quant_fund.models.modernism import bench_modernism
from quant_fund.models.postmodernism import bench_postmodernism
from quant_fund.models.renaissance_literature import bench_renaissance_literature
from quant_fund.models.romanticism import bench_romanticism
from quant_fund.models.victorian_studies import bench_victorian_studies

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


def bench_medieval_literature_family(seed: int = _SEED + 47300) -> dict[str, float]:
    return _finite_blob(bench_medieval_literature(seed))


def bench_renaissance_literature_family(seed: int = _SEED + 47301) -> dict[str, float]:
    return _finite_blob(bench_renaissance_literature(seed))


def bench_romanticism_family(seed: int = _SEED + 47302) -> dict[str, float]:
    return _finite_blob(bench_romanticism(seed))


def bench_modernism_family(seed: int = _SEED + 47303) -> dict[str, float]:
    return _finite_blob(bench_modernism(seed))


def bench_postmodernism_family(seed: int = _SEED + 47304) -> dict[str, float]:
    return _finite_blob(bench_postmodernism(seed))


def bench_victorian_studies_family(seed: int = _SEED + 47305) -> dict[str, float]:
    return _finite_blob(bench_victorian_studies(seed))
