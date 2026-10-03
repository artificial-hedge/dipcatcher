"""Wave-947 spectral-interlacing canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bezout_matrix import bench_bezout_matrix
from quant_fund.models.cauchy_interlace import bench_cauchy_interlace
from quant_fund.models.haynsworth_inertia import bench_haynsworth_inertia
from quant_fund.models.min_max_eig import bench_min_max_eig
from quant_fund.models.sturm_sequence import bench_sturm_sequence
from quant_fund.models.sylvester_law import bench_sylvester_law

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


def bench_cauchy_interlace_family(seed: int = _SEED + 33500) -> dict[str, float]:
    return _finite_blob(bench_cauchy_interlace(seed))


def bench_sylvester_law_family(seed: int = _SEED + 33501) -> dict[str, float]:
    return _finite_blob(bench_sylvester_law(seed))


def bench_haynsworth_inertia_family(seed: int = _SEED + 33502) -> dict[str, float]:
    return _finite_blob(bench_haynsworth_inertia(seed))


def bench_min_max_eig_family(seed: int = _SEED + 33503) -> dict[str, float]:
    return _finite_blob(bench_min_max_eig(seed))


def bench_sturm_sequence_family(seed: int = _SEED + 33504) -> dict[str, float]:
    return _finite_blob(bench_sturm_sequence(seed))


def bench_bezout_matrix_family(seed: int = _SEED + 33505) -> dict[str, float]:
    return _finite_blob(bench_bezout_matrix(seed))
