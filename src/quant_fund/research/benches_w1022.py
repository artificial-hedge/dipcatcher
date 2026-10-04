"""Wave-1022 economics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.auction_theory2 import bench_auction_theory2
from quant_fund.models.growth_theory import bench_growth_theory
from quant_fund.models.mechanism_design import bench_mechanism_design
from quant_fund.models.overlapping_gens import bench_overlapping_gens
from quant_fund.models.real_business import bench_real_business
from quant_fund.models.search_matching import bench_search_matching

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


def bench_growth_theory_family(seed: int = _SEED + 41000) -> dict[str, float]:
    return _finite_blob(bench_growth_theory(seed))


def bench_overlapping_gens_family(seed: int = _SEED + 41001) -> dict[str, float]:
    return _finite_blob(bench_overlapping_gens(seed))


def bench_real_business_family(seed: int = _SEED + 41002) -> dict[str, float]:
    return _finite_blob(bench_real_business(seed))


def bench_search_matching_family(seed: int = _SEED + 41003) -> dict[str, float]:
    return _finite_blob(bench_search_matching(seed))


def bench_mechanism_design_family(seed: int = _SEED + 41004) -> dict[str, float]:
    return _finite_blob(bench_mechanism_design(seed))


def bench_auction_theory2_family(seed: int = _SEED + 41005) -> dict[str, float]:
    return _finite_blob(bench_auction_theory2(seed))
