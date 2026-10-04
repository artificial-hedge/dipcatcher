"""Wave-1139 medicine-6 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.dentistry_2 import bench_dentistry_2
from quant_fund.models.dietetics import bench_dietetics
from quant_fund.models.occupational_therapy import bench_occupational_therapy
from quant_fund.models.optometry import bench_optometry
from quant_fund.models.physiotherapy import bench_physiotherapy
from quant_fund.models.podiatry import bench_podiatry

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


def bench_optometry_family(seed: int = _SEED + 52700) -> dict[str, float]:
    return _finite_blob(bench_optometry(seed))


def bench_dentistry_2_family(seed: int = _SEED + 52701) -> dict[str, float]:
    return _finite_blob(bench_dentistry_2(seed))


def bench_podiatry_family(seed: int = _SEED + 52702) -> dict[str, float]:
    return _finite_blob(bench_podiatry(seed))


def bench_dietetics_family(seed: int = _SEED + 52703) -> dict[str, float]:
    return _finite_blob(bench_dietetics(seed))


def bench_physiotherapy_family(seed: int = _SEED + 52704) -> dict[str, float]:
    return _finite_blob(bench_physiotherapy(seed))


def bench_occupational_therapy_family(seed: int = _SEED + 52705) -> dict[str, float]:
    return _finite_blob(bench_occupational_therapy(seed))
