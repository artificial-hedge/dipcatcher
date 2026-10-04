"""Wave-974 interpolation-theory canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.complex_interp import bench_complex_interp
from quant_fund.models.lorentz_space import bench_lorentz_space
from quant_fund.models.marcinkiewicz_interp import bench_marcinkiewicz_interp
from quant_fund.models.peetre_kfunctor import bench_peetre_kfunctor
from quant_fund.models.real_interp_k import bench_real_interp_k
from quant_fund.models.reiteration_thm import bench_reiteration_thm

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


def bench_real_interp_k_family(seed: int = _SEED + 36200) -> dict[str, float]:
    return _finite_blob(bench_real_interp_k(seed))


def bench_complex_interp_family(seed: int = _SEED + 36201) -> dict[str, float]:
    return _finite_blob(bench_complex_interp(seed))


def bench_lorentz_space_family(seed: int = _SEED + 36202) -> dict[str, float]:
    return _finite_blob(bench_lorentz_space(seed))


def bench_marcinkiewicz_interp_family(seed: int = _SEED + 36203) -> dict[str, float]:
    return _finite_blob(bench_marcinkiewicz_interp(seed))


def bench_peetre_kfunctor_family(seed: int = _SEED + 36204) -> dict[str, float]:
    return _finite_blob(bench_peetre_kfunctor(seed))


def bench_reiteration_thm_family(seed: int = _SEED + 36205) -> dict[str, float]:
    return _finite_blob(bench_reiteration_thm(seed))
