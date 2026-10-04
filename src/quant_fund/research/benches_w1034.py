"""Wave-1034 industrial-engineering canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.ergonomics import bench_ergonomics
from quant_fund.models.facility_layout import bench_facility_layout
from quant_fund.models.manufacturing_sys import bench_manufacturing_sys
from quant_fund.models.operations_research import bench_operations_research
from quant_fund.models.quality_control import bench_quality_control
from quant_fund.models.supply_chain import bench_supply_chain

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


def bench_operations_research_family(seed: int = _SEED + 42200) -> dict[str, float]:
    return _finite_blob(bench_operations_research(seed))


def bench_supply_chain_family(seed: int = _SEED + 42201) -> dict[str, float]:
    return _finite_blob(bench_supply_chain(seed))


def bench_manufacturing_sys_family(seed: int = _SEED + 42202) -> dict[str, float]:
    return _finite_blob(bench_manufacturing_sys(seed))


def bench_quality_control_family(seed: int = _SEED + 42203) -> dict[str, float]:
    return _finite_blob(bench_quality_control(seed))


def bench_ergonomics_family(seed: int = _SEED + 42204) -> dict[str, float]:
    return _finite_blob(bench_ergonomics(seed))


def bench_facility_layout_family(seed: int = _SEED + 42205) -> dict[str, float]:
    return _finite_blob(bench_facility_layout(seed))
