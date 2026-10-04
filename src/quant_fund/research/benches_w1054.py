"""Wave-1054 sociology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.criminology import bench_criminology
from quant_fund.models.demography import bench_demography
from quant_fund.models.economic_sociology import bench_economic_sociology
from quant_fund.models.social_networks import bench_social_networks
from quant_fund.models.social_stratification import bench_social_stratification
from quant_fund.models.urban_sociology import bench_urban_sociology

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


def bench_social_networks_family(seed: int = _SEED + 44200) -> dict[str, float]:
    return _finite_blob(bench_social_networks(seed))


def bench_demography_family(seed: int = _SEED + 44201) -> dict[str, float]:
    return _finite_blob(bench_demography(seed))


def bench_criminology_family(seed: int = _SEED + 44202) -> dict[str, float]:
    return _finite_blob(bench_criminology(seed))


def bench_urban_sociology_family(seed: int = _SEED + 44203) -> dict[str, float]:
    return _finite_blob(bench_urban_sociology(seed))


def bench_economic_sociology_family(seed: int = _SEED + 44204) -> dict[str, float]:
    return _finite_blob(bench_economic_sociology(seed))


def bench_social_stratification_family(seed: int = _SEED + 44205) -> dict[str, float]:
    return _finite_blob(bench_social_stratification(seed))
