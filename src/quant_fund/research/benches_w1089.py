"""Wave-1089 linguistics-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.computational_linguistics import bench_computational_linguistics
from quant_fund.models.corpus_linguistics import bench_corpus_linguistics
from quant_fund.models.dialectology import bench_dialectology
from quant_fund.models.historical_linguistics import bench_historical_linguistics
from quant_fund.models.psycholinguistics import bench_psycholinguistics
from quant_fund.models.sociolinguistics import bench_sociolinguistics

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


def bench_sociolinguistics_family(seed: int = _SEED + 47700) -> dict[str, float]:
    return _finite_blob(bench_sociolinguistics(seed))


def bench_psycholinguistics_family(seed: int = _SEED + 47701) -> dict[str, float]:
    return _finite_blob(bench_psycholinguistics(seed))


def bench_computational_linguistics_family(seed: int = _SEED + 47702) -> dict[str, float]:
    return _finite_blob(bench_computational_linguistics(seed))


def bench_corpus_linguistics_family(seed: int = _SEED + 47703) -> dict[str, float]:
    return _finite_blob(bench_corpus_linguistics(seed))


def bench_dialectology_family(seed: int = _SEED + 47704) -> dict[str, float]:
    return _finite_blob(bench_dialectology(seed))


def bench_historical_linguistics_family(seed: int = _SEED + 47705) -> dict[str, float]:
    return _finite_blob(bench_historical_linguistics(seed))
