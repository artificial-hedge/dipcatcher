"""Wave-1163 business canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.accounting_2 import bench_accounting_2
from quant_fund.models.business_administration import bench_business_administration
from quant_fund.models.entrepreneurship_2 import bench_entrepreneurship_2
from quant_fund.models.finance_4 import bench_finance_4
from quant_fund.models.management_2 import bench_management_2
from quant_fund.models.marketing_2 import bench_marketing_2

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


def bench_accounting_2_family(seed: int = _SEED + 55100) -> dict[str, float]:
    return _finite_blob(bench_accounting_2(seed))


def bench_finance_4_family(seed: int = _SEED + 55101) -> dict[str, float]:
    return _finite_blob(bench_finance_4(seed))


def bench_marketing_2_family(seed: int = _SEED + 55102) -> dict[str, float]:
    return _finite_blob(bench_marketing_2(seed))


def bench_management_2_family(seed: int = _SEED + 55103) -> dict[str, float]:
    return _finite_blob(bench_management_2(seed))


def bench_entrepreneurship_2_family(seed: int = _SEED + 55104) -> dict[str, float]:
    return _finite_blob(bench_entrepreneurship_2(seed))


def bench_business_administration_family(seed: int = _SEED + 55105) -> dict[str, float]:
    return _finite_blob(bench_business_administration(seed))
