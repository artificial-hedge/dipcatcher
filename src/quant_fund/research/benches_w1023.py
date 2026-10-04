"""Wave-1023 finance-theory canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.arbitrage_pricing import bench_arbitrage_pricing
from quant_fund.models.black_scholes import bench_black_scholes
from quant_fund.models.capm_model import bench_capm_model
from quant_fund.models.corporate_finance import bench_corporate_finance
from quant_fund.models.default_risk import bench_default_risk
from quant_fund.models.yield_curve import bench_yield_curve

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


def bench_capm_model_family(seed: int = _SEED + 41100) -> dict[str, float]:
    return _finite_blob(bench_capm_model(seed))


def bench_arbitrage_pricing_family(seed: int = _SEED + 41101) -> dict[str, float]:
    return _finite_blob(bench_arbitrage_pricing(seed))


def bench_black_scholes_family(seed: int = _SEED + 41102) -> dict[str, float]:
    return _finite_blob(bench_black_scholes(seed))


def bench_yield_curve_family(seed: int = _SEED + 41103) -> dict[str, float]:
    return _finite_blob(bench_yield_curve(seed))


def bench_default_risk_family(seed: int = _SEED + 41104) -> dict[str, float]:
    return _finite_blob(bench_default_risk(seed))


def bench_corporate_finance_family(seed: int = _SEED + 41105) -> dict[str, float]:
    return _finite_blob(bench_corporate_finance(seed))
