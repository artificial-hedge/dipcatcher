"""Wave-1150 microbial-genetics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bacteriology import bench_bacteriology
from quant_fund.models.epigenetics import bench_epigenetics
from quant_fund.models.immunogenetics import bench_immunogenetics
from quant_fund.models.microbiology_2 import bench_microbiology_2
from quant_fund.models.molecular_genetics import bench_molecular_genetics
from quant_fund.models.virology_2 import bench_virology_2

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


def bench_microbiology_2_family(seed: int = _SEED + 53800) -> dict[str, float]:
    return _finite_blob(bench_microbiology_2(seed))


def bench_bacteriology_family(seed: int = _SEED + 53801) -> dict[str, float]:
    return _finite_blob(bench_bacteriology(seed))


def bench_virology_2_family(seed: int = _SEED + 53802) -> dict[str, float]:
    return _finite_blob(bench_virology_2(seed))


def bench_immunogenetics_family(seed: int = _SEED + 53803) -> dict[str, float]:
    return _finite_blob(bench_immunogenetics(seed))


def bench_molecular_genetics_family(seed: int = _SEED + 53804) -> dict[str, float]:
    return _finite_blob(bench_molecular_genetics(seed))


def bench_epigenetics_family(seed: int = _SEED + 53805) -> dict[str, float]:
    return _finite_blob(bench_epigenetics(seed))
