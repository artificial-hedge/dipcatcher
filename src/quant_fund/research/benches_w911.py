"""Wave-911 computational-geometry-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bezier_eval import bench_bezier_eval
from quant_fund.models.chan_hull import bench_chan_hull
from quant_fund.models.cohen_sutherland import bench_cohen_sutherland
from quant_fund.models.gift_wrap import bench_gift_wrap
from quant_fund.models.liang_barsky import bench_liang_barsky
from quant_fund.models.monotone_chain import bench_monotone_chain

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


def bench_monotone_chain_family(seed: int = _SEED + 29900) -> dict[str, float]:
    return _finite_blob(bench_monotone_chain(seed))


def bench_gift_wrap_family(seed: int = _SEED + 29901) -> dict[str, float]:
    return _finite_blob(bench_gift_wrap(seed))


def bench_chan_hull_family(seed: int = _SEED + 29902) -> dict[str, float]:
    return _finite_blob(bench_chan_hull(seed))


def bench_liang_barsky_family(seed: int = _SEED + 29903) -> dict[str, float]:
    return _finite_blob(bench_liang_barsky(seed))


def bench_cohen_sutherland_family(seed: int = _SEED + 29904) -> dict[str, float]:
    return _finite_blob(bench_cohen_sutherland(seed))


def bench_bezier_eval_family(seed: int = _SEED + 29905) -> dict[str, float]:
    return _finite_blob(bench_bezier_eval(seed))
