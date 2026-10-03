"""Wave-991 homogenization canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bloch_decomp import bench_bloch_decomp
from quant_fund.models.gamma_convergence import bench_gamma_convergence
from quant_fund.models.h_convergence import bench_h_convergence
from quant_fund.models.homogenization import bench_homogenization
from quant_fund.models.mosco_conv import bench_mosco_conv
from quant_fund.models.two_scale_conv import bench_two_scale_conv

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


def bench_homogenization_family(seed: int = _SEED + 37900) -> dict[str, float]:
    return _finite_blob(bench_homogenization(seed))


def bench_two_scale_conv_family(seed: int = _SEED + 37901) -> dict[str, float]:
    return _finite_blob(bench_two_scale_conv(seed))


def bench_gamma_convergence_family(seed: int = _SEED + 37902) -> dict[str, float]:
    return _finite_blob(bench_gamma_convergence(seed))


def bench_mosco_conv_family(seed: int = _SEED + 37903) -> dict[str, float]:
    return _finite_blob(bench_mosco_conv(seed))


def bench_bloch_decomp_family(seed: int = _SEED + 37904) -> dict[str, float]:
    return _finite_blob(bench_bloch_decomp(seed))


def bench_h_convergence_family(seed: int = _SEED + 37905) -> dict[str, float]:
    return _finite_blob(bench_h_convergence(seed))
