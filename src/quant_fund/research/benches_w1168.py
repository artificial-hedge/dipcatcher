"""Wave-1168 logistics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.aviation_2 import bench_aviation_2
from quant_fund.models.logistics_2 import bench_logistics_2
from quant_fund.models.maritime_studies_2 import bench_maritime_studies_2
from quant_fund.models.supply_chain_2 import bench_supply_chain_2
from quant_fund.models.transportation_2 import bench_transportation_2
from quant_fund.models.warehousing_2 import bench_warehousing_2

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


def bench_transportation_2_family(seed: int = _SEED + 55600) -> dict[str, float]:
    return _finite_blob(bench_transportation_2(seed))


def bench_logistics_2_family(seed: int = _SEED + 55601) -> dict[str, float]:
    return _finite_blob(bench_logistics_2(seed))


def bench_supply_chain_2_family(seed: int = _SEED + 55602) -> dict[str, float]:
    return _finite_blob(bench_supply_chain_2(seed))


def bench_warehousing_2_family(seed: int = _SEED + 55603) -> dict[str, float]:
    return _finite_blob(bench_warehousing_2(seed))


def bench_maritime_studies_2_family(seed: int = _SEED + 55604) -> dict[str, float]:
    return _finite_blob(bench_maritime_studies_2(seed))


def bench_aviation_2_family(seed: int = _SEED + 55605) -> dict[str, float]:
    return _finite_blob(bench_aviation_2(seed))
