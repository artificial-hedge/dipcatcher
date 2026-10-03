"""Wave-963 spectral-theory-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.atkinson_thm import bench_atkinson_thm
from quant_fund.models.browder_operator import bench_browder_operator
from quant_fund.models.essential_spectrum import bench_essential_spectrum
from quant_fund.models.fredholm_index import bench_fredholm_index
from quant_fund.models.riesz_schauder import bench_riesz_schauder
from quant_fund.models.weyl_theorem import bench_weyl_theorem

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


def bench_fredholm_index_family(seed: int = _SEED + 35100) -> dict[str, float]:
    return _finite_blob(bench_fredholm_index(seed))


def bench_weyl_theorem_family(seed: int = _SEED + 35101) -> dict[str, float]:
    return _finite_blob(bench_weyl_theorem(seed))


def bench_essential_spectrum_family(seed: int = _SEED + 35102) -> dict[str, float]:
    return _finite_blob(bench_essential_spectrum(seed))


def bench_browder_operator_family(seed: int = _SEED + 35103) -> dict[str, float]:
    return _finite_blob(bench_browder_operator(seed))


def bench_riesz_schauder_family(seed: int = _SEED + 35104) -> dict[str, float]:
    return _finite_blob(bench_riesz_schauder(seed))


def bench_atkinson_thm_family(seed: int = _SEED + 35105) -> dict[str, float]:
    return _finite_blob(bench_atkinson_thm(seed))
