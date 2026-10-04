"""Wave-1090 history-of-science canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.history_of_science import bench_history_of_science
from quant_fund.models.information_history import bench_information_history
from quant_fund.models.media_archaeology import bench_media_archaeology
from quant_fund.models.philosophy_of_technology import bench_philosophy_of_technology
from quant_fund.models.sts_studies import bench_sts_studies
from quant_fund.models.technology_studies import bench_technology_studies

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


def bench_history_of_science_family(seed: int = _SEED + 47800) -> dict[str, float]:
    return _finite_blob(bench_history_of_science(seed))


def bench_sts_studies_family(seed: int = _SEED + 47801) -> dict[str, float]:
    return _finite_blob(bench_sts_studies(seed))


def bench_philosophy_of_technology_family(seed: int = _SEED + 47802) -> dict[str, float]:
    return _finite_blob(bench_philosophy_of_technology(seed))


def bench_media_archaeology_family(seed: int = _SEED + 47803) -> dict[str, float]:
    return _finite_blob(bench_media_archaeology(seed))


def bench_information_history_family(seed: int = _SEED + 47804) -> dict[str, float]:
    return _finite_blob(bench_information_history(seed))


def bench_technology_studies_family(seed: int = _SEED + 47805) -> dict[str, float]:
    return _finite_blob(bench_technology_studies(seed))
