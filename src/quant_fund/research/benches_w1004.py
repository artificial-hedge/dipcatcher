"""Wave-1004 kinetic-theory canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bgk_model import bench_bgk_model
from quant_fund.models.boltzmann_eq import bench_boltzmann_eq
from quant_fund.models.chapman_enskog import bench_chapman_enskog
from quant_fund.models.h_theorem import bench_h_theorem
from quant_fund.models.landau_damping import bench_landau_damping
from quant_fund.models.vlasov_eq import bench_vlasov_eq

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


def bench_boltzmann_eq_family(seed: int = _SEED + 39200) -> dict[str, float]:
    return _finite_blob(bench_boltzmann_eq(seed))


def bench_vlasov_eq_family(seed: int = _SEED + 39201) -> dict[str, float]:
    return _finite_blob(bench_vlasov_eq(seed))


def bench_bgk_model_family(seed: int = _SEED + 39202) -> dict[str, float]:
    return _finite_blob(bench_bgk_model(seed))


def bench_chapman_enskog_family(seed: int = _SEED + 39203) -> dict[str, float]:
    return _finite_blob(bench_chapman_enskog(seed))


def bench_h_theorem_family(seed: int = _SEED + 39204) -> dict[str, float]:
    return _finite_blob(bench_h_theorem(seed))


def bench_landau_damping_family(seed: int = _SEED + 39205) -> dict[str, float]:
    return _finite_blob(bench_landau_damping(seed))
