"""Wave-940 variational-inequality canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.forward_reflected import bench_forward_reflected
from quant_fund.models.korpelevich_eg import bench_korpelevich_eg
from quant_fund.models.popov_alg import bench_popov_alg
from quant_fund.models.reflected_golden import bench_reflected_golden
from quant_fund.models.subgradient_extragradient import bench_subgradient_extragradient
from quant_fund.models.tseng_fb import bench_tseng_fb

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


def bench_subgradient_extragradient_family(seed: int = _SEED + 32800) -> dict[str, float]:
    return _finite_blob(bench_subgradient_extragradient(seed))


def bench_korpelevich_eg_family(seed: int = _SEED + 32801) -> dict[str, float]:
    return _finite_blob(bench_korpelevich_eg(seed))


def bench_popov_alg_family(seed: int = _SEED + 32802) -> dict[str, float]:
    return _finite_blob(bench_popov_alg(seed))


def bench_tseng_fb_family(seed: int = _SEED + 32803) -> dict[str, float]:
    return _finite_blob(bench_tseng_fb(seed))


def bench_forward_reflected_family(seed: int = _SEED + 32804) -> dict[str, float]:
    return _finite_blob(bench_forward_reflected(seed))


def bench_reflected_golden_family(seed: int = _SEED + 32805) -> dict[str, float]:
    return _finite_blob(bench_reflected_golden(seed))
