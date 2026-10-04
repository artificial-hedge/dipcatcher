"""Wave-989 microlocal-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.fbi_transform import bench_fbi_transform
from quant_fund.models.melrose_bdy import bench_melrose_bdy
from quant_fund.models.parametrix import bench_parametrix
from quant_fund.models.propagation_thm import bench_propagation_thm
from quant_fund.models.sg_calculus import bench_sg_calculus
from quant_fund.models.wave_eq_group import bench_wave_eq_group

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


def bench_parametrix_family(seed: int = _SEED + 37700) -> dict[str, float]:
    return _finite_blob(bench_parametrix(seed))


def bench_wave_eq_group_family(seed: int = _SEED + 37701) -> dict[str, float]:
    return _finite_blob(bench_wave_eq_group(seed))


def bench_propagation_thm_family(seed: int = _SEED + 37702) -> dict[str, float]:
    return _finite_blob(bench_propagation_thm(seed))


def bench_melrose_bdy_family(seed: int = _SEED + 37703) -> dict[str, float]:
    return _finite_blob(bench_melrose_bdy(seed))


def bench_fbi_transform_family(seed: int = _SEED + 37704) -> dict[str, float]:
    return _finite_blob(bench_fbi_transform(seed))


def bench_sg_calculus_family(seed: int = _SEED + 37705) -> dict[str, float]:
    return _finite_blob(bench_sg_calculus(seed))
