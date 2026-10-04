"""Wave-1172 business canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.accounting_3 import bench_accounting_3
from quant_fund.models.entrepreneurship_3 import bench_entrepreneurship_3
from quant_fund.models.finance_5 import bench_finance_5
from quant_fund.models.management_3 import bench_management_3
from quant_fund.models.marketing_3 import bench_marketing_3
from quant_fund.models.organizational_behavior import bench_organizational_behavior

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


def bench_management_3_family(seed: int = _SEED + 56000) -> dict[str, float]:
    return _finite_blob(bench_management_3(seed))


def bench_marketing_3_family(seed: int = _SEED + 56001) -> dict[str, float]:
    return _finite_blob(bench_marketing_3(seed))


def bench_accounting_3_family(seed: int = _SEED + 56002) -> dict[str, float]:
    return _finite_blob(bench_accounting_3(seed))


def bench_finance_5_family(seed: int = _SEED + 56003) -> dict[str, float]:
    return _finite_blob(bench_finance_5(seed))


def bench_entrepreneurship_3_family(seed: int = _SEED + 56004) -> dict[str, float]:
    return _finite_blob(bench_entrepreneurship_3(seed))


def bench_organizational_behavior_family(seed: int = _SEED + 56005) -> dict[str, float]:
    return _finite_blob(bench_organizational_behavior(seed))
