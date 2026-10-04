"""Wave-1012 nuclear/particle-physics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bcs_theory import bench_bcs_theory
from quant_fund.models.cabibbo_km import bench_cabibbo_km
from quant_fund.models.nuclear_liquid_drop import bench_nuclear_liquid_drop
from quant_fund.models.nuclear_shell_model import bench_nuclear_shell_model
from quant_fund.models.parton_model import bench_parton_model
from quant_fund.models.quark_model import bench_quark_model

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


def bench_bcs_theory_family(seed: int = _SEED + 40000) -> dict[str, float]:
    return _finite_blob(bench_bcs_theory(seed))


def bench_nuclear_shell_model_family(seed: int = _SEED + 40001) -> dict[str, float]:
    return _finite_blob(bench_nuclear_shell_model(seed))


def bench_nuclear_liquid_drop_family(seed: int = _SEED + 40002) -> dict[str, float]:
    return _finite_blob(bench_nuclear_liquid_drop(seed))


def bench_quark_model_family(seed: int = _SEED + 40003) -> dict[str, float]:
    return _finite_blob(bench_quark_model(seed))


def bench_parton_model_family(seed: int = _SEED + 40004) -> dict[str, float]:
    return _finite_blob(bench_parton_model(seed))


def bench_cabibbo_km_family(seed: int = _SEED + 40005) -> dict[str, float]:
    return _finite_blob(bench_cabibbo_km(seed))
