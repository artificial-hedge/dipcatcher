"""Wave-1187 fashion canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.apparel_studies import bench_apparel_studies
from quant_fund.models.costume_design import bench_costume_design
from quant_fund.models.fashion_studies import bench_fashion_studies
from quant_fund.models.footwear_design import bench_footwear_design
from quant_fund.models.jewelry_design import bench_jewelry_design
from quant_fund.models.textile_studies import bench_textile_studies

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


def bench_fashion_studies_family(seed: int = _SEED + 57500) -> dict[str, float]:
    return _finite_blob(bench_fashion_studies(seed))


def bench_textile_studies_family(seed: int = _SEED + 57501) -> dict[str, float]:
    return _finite_blob(bench_textile_studies(seed))


def bench_costume_design_family(seed: int = _SEED + 57502) -> dict[str, float]:
    return _finite_blob(bench_costume_design(seed))


def bench_jewelry_design_family(seed: int = _SEED + 57503) -> dict[str, float]:
    return _finite_blob(bench_jewelry_design(seed))


def bench_footwear_design_family(seed: int = _SEED + 57504) -> dict[str, float]:
    return _finite_blob(bench_footwear_design(seed))


def bench_apparel_studies_family(seed: int = _SEED + 57505) -> dict[str, float]:
    return _finite_blob(bench_apparel_studies(seed))
