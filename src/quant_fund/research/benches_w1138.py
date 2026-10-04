"""Wave-1138 economics-5 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.behavioral_economics import bench_behavioral_economics
from quant_fund.models.econ_neuroscience import bench_econ_neuroscience
from quant_fund.models.evolutionary_economics import bench_evolutionary_economics
from quant_fund.models.experimental_economics_2 import bench_experimental_economics_2
from quant_fund.models.institutional_economics import bench_institutional_economics
from quant_fund.models.political_economy_2 import bench_political_economy_2

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


def bench_behavioral_economics_family(seed: int = _SEED + 52600) -> dict[str, float]:
    return _finite_blob(bench_behavioral_economics(seed))


def bench_econ_neuroscience_family(seed: int = _SEED + 52601) -> dict[str, float]:
    return _finite_blob(bench_econ_neuroscience(seed))


def bench_experimental_economics_2_family(seed: int = _SEED + 52602) -> dict[str, float]:
    return _finite_blob(bench_experimental_economics_2(seed))


def bench_institutional_economics_family(seed: int = _SEED + 52603) -> dict[str, float]:
    return _finite_blob(bench_institutional_economics(seed))


def bench_evolutionary_economics_family(seed: int = _SEED + 52604) -> dict[str, float]:
    return _finite_blob(bench_evolutionary_economics(seed))


def bench_political_economy_2_family(seed: int = _SEED + 52605) -> dict[str, float]:
    return _finite_blob(bench_political_economy_2(seed))
