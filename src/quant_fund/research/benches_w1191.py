"""Wave-1191 media canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.advertising_studies import bench_advertising_studies
from quant_fund.models.broadcasting_studies import bench_broadcasting_studies
from quant_fund.models.journalism_studies import bench_journalism_studies
from quant_fund.models.news_media import bench_news_media
from quant_fund.models.public_relations_studies import bench_public_relations_studies
from quant_fund.models.publishing_studies import bench_publishing_studies

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


def bench_journalism_studies_family(seed: int = _SEED + 57900) -> dict[str, float]:
    return _finite_blob(bench_journalism_studies(seed))


def bench_advertising_studies_family(seed: int = _SEED + 57901) -> dict[str, float]:
    return _finite_blob(bench_advertising_studies(seed))


def bench_broadcasting_studies_family(seed: int = _SEED + 57902) -> dict[str, float]:
    return _finite_blob(bench_broadcasting_studies(seed))


def bench_news_media_family(seed: int = _SEED + 57903) -> dict[str, float]:
    return _finite_blob(bench_news_media(seed))


def bench_public_relations_studies_family(seed: int = _SEED + 57904) -> dict[str, float]:
    return _finite_blob(bench_public_relations_studies(seed))


def bench_publishing_studies_family(seed: int = _SEED + 57905) -> dict[str, float]:
    return _finite_blob(bench_publishing_studies(seed))
