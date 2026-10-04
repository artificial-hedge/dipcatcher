"""Wave-1115 economics-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.financial_economics import bench_financial_economics
from quant_fund.models.industrial_organization import bench_industrial_organization
from quant_fund.models.international_economics import bench_international_economics
from quant_fund.models.labor_economics import bench_labor_economics
from quant_fund.models.monetary_economics import bench_monetary_economics
from quant_fund.models.public_economics import bench_public_economics

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


def bench_labor_economics_family(seed: int = _SEED + 50300) -> dict[str, float]:
    return _finite_blob(bench_labor_economics(seed))


def bench_public_economics_family(seed: int = _SEED + 50301) -> dict[str, float]:
    return _finite_blob(bench_public_economics(seed))


def bench_industrial_organization_family(seed: int = _SEED + 50302) -> dict[str, float]:
    return _finite_blob(bench_industrial_organization(seed))


def bench_international_economics_family(seed: int = _SEED + 50303) -> dict[str, float]:
    return _finite_blob(bench_international_economics(seed))


def bench_financial_economics_family(seed: int = _SEED + 50304) -> dict[str, float]:
    return _finite_blob(bench_financial_economics(seed))


def bench_monetary_economics_family(seed: int = _SEED + 50305) -> dict[str, float]:
    return _finite_blob(bench_monetary_economics(seed))
