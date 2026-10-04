"""Wave-967 C*-dynamics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.crossed_product import bench_crossed_product
from quant_fund.models.cstar_dynamics import bench_cstar_dynamics
from quant_fund.models.kirchberg_absorb import bench_kirchberg_absorb
from quant_fund.models.rokhlin_action import bench_rokhlin_action
from quant_fund.models.taf_dim import bench_taf_dim
from quant_fund.models.z_stability import bench_z_stability

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


def bench_cstar_dynamics_family(seed: int = _SEED + 35500) -> dict[str, float]:
    return _finite_blob(bench_cstar_dynamics(seed))


def bench_crossed_product_family(seed: int = _SEED + 35501) -> dict[str, float]:
    return _finite_blob(bench_crossed_product(seed))


def bench_rokhlin_action_family(seed: int = _SEED + 35502) -> dict[str, float]:
    return _finite_blob(bench_rokhlin_action(seed))


def bench_kirchberg_absorb_family(seed: int = _SEED + 35503) -> dict[str, float]:
    return _finite_blob(bench_kirchberg_absorb(seed))


def bench_taf_dim_family(seed: int = _SEED + 35504) -> dict[str, float]:
    return _finite_blob(bench_taf_dim(seed))


def bench_z_stability_family(seed: int = _SEED + 35505) -> dict[str, float]:
    return _finite_blob(bench_z_stability(seed))
