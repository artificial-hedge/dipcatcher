"""Wave-926 information-geometry-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.alpha_divergence import bench_alpha_divergence
from quant_fund.models.amari_connection import bench_amari_connection
from quant_fund.models.csiszar_div import bench_csiszar_div
from quant_fund.models.dual_connection import bench_dual_connection
from quant_fund.models.f_divergence import bench_f_divergence
from quant_fund.models.tsallis_entropy import bench_tsallis_entropy

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


def bench_f_divergence_family(seed: int = _SEED + 31400) -> dict[str, float]:
    return _finite_blob(bench_f_divergence(seed))


def bench_alpha_divergence_family(seed: int = _SEED + 31401) -> dict[str, float]:
    return _finite_blob(bench_alpha_divergence(seed))


def bench_csiszar_div_family(seed: int = _SEED + 31402) -> dict[str, float]:
    return _finite_blob(bench_csiszar_div(seed))


def bench_amari_connection_family(seed: int = _SEED + 31403) -> dict[str, float]:
    return _finite_blob(bench_amari_connection(seed))


def bench_dual_connection_family(seed: int = _SEED + 31404) -> dict[str, float]:
    return _finite_blob(bench_dual_connection(seed))


def bench_tsallis_entropy_family(seed: int = _SEED + 31405) -> dict[str, float]:
    return _finite_blob(bench_tsallis_entropy(seed))
