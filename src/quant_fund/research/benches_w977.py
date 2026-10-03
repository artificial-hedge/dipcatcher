"""Wave-977 Bochner/vector-valued canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bochner_integral import bench_bochner_integral
from quant_fund.models.bochner_meas import bench_bochner_meas
from quant_fund.models.lusin_rep import bench_lusin_rep
from quant_fund.models.norm_integrable import bench_norm_integrable
from quant_fund.models.pettis_weak import bench_pettis_weak
from quant_fund.models.radon_nikodym_prop import bench_radon_nikodym_prop

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


def bench_bochner_integral_family(seed: int = _SEED + 36500) -> dict[str, float]:
    return _finite_blob(bench_bochner_integral(seed))


def bench_lusin_rep_family(seed: int = _SEED + 36501) -> dict[str, float]:
    return _finite_blob(bench_lusin_rep(seed))


def bench_radon_nikodym_prop_family(seed: int = _SEED + 36502) -> dict[str, float]:
    return _finite_blob(bench_radon_nikodym_prop(seed))


def bench_bochner_meas_family(seed: int = _SEED + 36503) -> dict[str, float]:
    return _finite_blob(bench_bochner_meas(seed))


def bench_norm_integrable_family(seed: int = _SEED + 36504) -> dict[str, float]:
    return _finite_blob(bench_norm_integrable(seed))


def bench_pettis_weak_family(seed: int = _SEED + 36505) -> dict[str, float]:
    return _finite_blob(bench_pettis_weak(seed))
