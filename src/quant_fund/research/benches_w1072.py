"""Wave-1072 film studies canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.cinema_studies import bench_cinema_studies
from quant_fund.models.documentary_studies import bench_documentary_studies
from quant_fund.models.film_history import bench_film_history
from quant_fund.models.film_studies import bench_film_studies
from quant_fund.models.film_theory import bench_film_theory
from quant_fund.models.screenwriting import bench_screenwriting

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


def bench_film_studies_family(seed: int = _SEED + 46000) -> dict[str, float]:
    return _finite_blob(bench_film_studies(seed))


def bench_cinema_studies_family(seed: int = _SEED + 46001) -> dict[str, float]:
    return _finite_blob(bench_cinema_studies(seed))


def bench_film_theory_family(seed: int = _SEED + 46002) -> dict[str, float]:
    return _finite_blob(bench_film_theory(seed))


def bench_film_history_family(seed: int = _SEED + 46003) -> dict[str, float]:
    return _finite_blob(bench_film_history(seed))


def bench_documentary_studies_family(seed: int = _SEED + 46004) -> dict[str, float]:
    return _finite_blob(bench_documentary_studies(seed))


def bench_screenwriting_family(seed: int = _SEED + 46005) -> dict[str, float]:
    return _finite_blob(bench_screenwriting(seed))
