"""Wave-938 fixed-point canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.averaged_operator import bench_averaged_operator
from quant_fund.models.cocoercive import bench_cocoercive
from quant_fund.models.fejer_monotone import bench_fejer_monotone
from quant_fund.models.firmly_nonexpansive import bench_firmly_nonexpansive
from quant_fund.models.monotone_inclusion import bench_monotone_inclusion
from quant_fund.models.quasinonexpansive import bench_quasinonexpansive

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


def bench_fejer_monotone_family(seed: int = _SEED + 32600) -> dict[str, float]:
    return _finite_blob(bench_fejer_monotone(seed))


def bench_firmly_nonexpansive_family(seed: int = _SEED + 32601) -> dict[str, float]:
    return _finite_blob(bench_firmly_nonexpansive(seed))


def bench_averaged_operator_family(seed: int = _SEED + 32602) -> dict[str, float]:
    return _finite_blob(bench_averaged_operator(seed))


def bench_cocoercive_family(seed: int = _SEED + 32603) -> dict[str, float]:
    return _finite_blob(bench_cocoercive(seed))


def bench_quasinonexpansive_family(seed: int = _SEED + 32604) -> dict[str, float]:
    return _finite_blob(bench_quasinonexpansive(seed))


def bench_monotone_inclusion_family(seed: int = _SEED + 32605) -> dict[str, float]:
    return _finite_blob(bench_monotone_inclusion(seed))
