"""Wave-1081 renaissance/early modern canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.baroque_studies import bench_baroque_studies
from quant_fund.models.early_modern import bench_early_modern
from quant_fund.models.enlightenment_studies import bench_enlightenment_studies
from quant_fund.models.humanism import bench_humanism
from quant_fund.models.reformation_studies import bench_reformation_studies
from quant_fund.models.renaissance_studies import bench_renaissance_studies

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


def bench_renaissance_studies_family(seed: int = _SEED + 46900) -> dict[str, float]:
    return _finite_blob(bench_renaissance_studies(seed))


def bench_early_modern_family(seed: int = _SEED + 46901) -> dict[str, float]:
    return _finite_blob(bench_early_modern(seed))


def bench_humanism_family(seed: int = _SEED + 46902) -> dict[str, float]:
    return _finite_blob(bench_humanism(seed))


def bench_reformation_studies_family(seed: int = _SEED + 46903) -> dict[str, float]:
    return _finite_blob(bench_reformation_studies(seed))


def bench_baroque_studies_family(seed: int = _SEED + 46904) -> dict[str, float]:
    return _finite_blob(bench_baroque_studies(seed))


def bench_enlightenment_studies_family(seed: int = _SEED + 46905) -> dict[str, float]:
    return _finite_blob(bench_enlightenment_studies(seed))
