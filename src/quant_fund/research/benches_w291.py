"""Wave-291 VLSI/EDA canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.a_star_route import bench_a_star_route
from quant_fund.models.drc_check import bench_drc_check
from quant_fund.models.levelize import bench_levelize
from quant_fund.models.netlist_parse import bench_netlist_parse
from quant_fund.models.place_quadratic import bench_place_quadratic
from quant_fund.models.sta_timing import bench_sta_timing

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_netlist_parse_family(seed: int = _SEED + 1652) -> dict[str, float]:
    return _floats(_finite_blob("netlist_parse", bench_netlist_parse(seed)))


def bench_sta_timing_family(seed: int = _SEED + 1653) -> dict[str, float]:
    return _floats(_finite_blob("sta_timing", bench_sta_timing(seed)))


def bench_a_star_route_family(seed: int = _SEED + 1654) -> dict[str, float]:
    return _floats(_finite_blob("a_star_route", bench_a_star_route(seed)))


def bench_drc_check_family(seed: int = _SEED + 1655) -> dict[str, float]:
    return _floats(_finite_blob("drc_check", bench_drc_check(seed)))


def bench_place_quadratic_family(seed: int = _SEED + 1656) -> dict[str, float]:
    return _floats(_finite_blob("place_quadratic", bench_place_quadratic(seed)))


def bench_levelize_family(seed: int = _SEED + 1657) -> dict[str, float]:
    return _floats(_finite_blob("levelize", bench_levelize(seed)))
