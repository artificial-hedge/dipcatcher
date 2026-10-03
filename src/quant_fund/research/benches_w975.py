"""Wave-975 distribution-theory canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.dist_convolution import bench_dist_convolution
from quant_fund.models.paley_wiener import bench_paley_wiener
from quant_fund.models.schwartz_dist import bench_schwartz_dist
from quant_fund.models.sing_support import bench_sing_support
from quant_fund.models.sobolev_trace import bench_sobolev_trace
from quant_fund.models.temper_dist import bench_temper_dist

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


def bench_schwartz_dist_family(seed: int = _SEED + 36300) -> dict[str, float]:
    return _finite_blob(bench_schwartz_dist(seed))


def bench_temper_dist_family(seed: int = _SEED + 36301) -> dict[str, float]:
    return _finite_blob(bench_temper_dist(seed))


def bench_dist_convolution_family(seed: int = _SEED + 36302) -> dict[str, float]:
    return _finite_blob(bench_dist_convolution(seed))


def bench_sing_support_family(seed: int = _SEED + 36303) -> dict[str, float]:
    return _finite_blob(bench_sing_support(seed))


def bench_paley_wiener_family(seed: int = _SEED + 36304) -> dict[str, float]:
    return _finite_blob(bench_paley_wiener(seed))


def bench_sobolev_trace_family(seed: int = _SEED + 36305) -> dict[str, float]:
    return _finite_blob(bench_sobolev_trace(seed))
