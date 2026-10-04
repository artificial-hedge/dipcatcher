"""Wave-1132 history-4 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.history_of_capitalism import bench_history_of_capitalism
from quant_fund.models.history_of_emotions import bench_history_of_emotions
from quant_fund.models.history_of_religions import bench_history_of_religions
from quant_fund.models.history_of_sexuality import bench_history_of_sexuality
from quant_fund.models.history_of_the_book import bench_history_of_the_book
from quant_fund.models.microhistory import bench_microhistory

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


def bench_history_of_emotions_family(seed: int = _SEED + 52000) -> dict[str, float]:
    return _finite_blob(bench_history_of_emotions(seed))


def bench_history_of_sexuality_family(seed: int = _SEED + 52001) -> dict[str, float]:
    return _finite_blob(bench_history_of_sexuality(seed))


def bench_history_of_the_book_family(seed: int = _SEED + 52002) -> dict[str, float]:
    return _finite_blob(bench_history_of_the_book(seed))


def bench_history_of_capitalism_family(seed: int = _SEED + 52003) -> dict[str, float]:
    return _finite_blob(bench_history_of_capitalism(seed))


def bench_history_of_religions_family(seed: int = _SEED + 52004) -> dict[str, float]:
    return _finite_blob(bench_history_of_religions(seed))


def bench_microhistory_family(seed: int = _SEED + 52005) -> dict[str, float]:
    return _finite_blob(bench_microhistory(seed))
