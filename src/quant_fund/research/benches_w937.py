"""Wave-937 operator-splitting canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.chambolle_pock import bench_chambolle_pock
from quant_fund.models.davis_yin import bench_davis_yin
from quant_fund.models.douglas_rachford import bench_douglas_rachford
from quant_fund.models.forward_backward import bench_forward_backward
from quant_fund.models.peaceman_rachford import bench_peaceman_rachford
from quant_fund.models.tseng_split import bench_tseng_split

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


def bench_douglas_rachford_family(seed: int = _SEED + 32500) -> dict[str, float]:
    return _finite_blob(bench_douglas_rachford(seed))


def bench_peaceman_rachford_family(seed: int = _SEED + 32501) -> dict[str, float]:
    return _finite_blob(bench_peaceman_rachford(seed))


def bench_tseng_split_family(seed: int = _SEED + 32502) -> dict[str, float]:
    return _finite_blob(bench_tseng_split(seed))


def bench_forward_backward_family(seed: int = _SEED + 32503) -> dict[str, float]:
    return _finite_blob(bench_forward_backward(seed))


def bench_chambolle_pock_family(seed: int = _SEED + 32504) -> dict[str, float]:
    return _finite_blob(bench_chambolle_pock(seed))


def bench_davis_yin_family(seed: int = _SEED + 32505) -> dict[str, float]:
    return _finite_blob(bench_davis_yin(seed))
