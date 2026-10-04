"""Wave-1122 archaeology-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.archaeogenetics import bench_archaeogenetics
from quant_fund.models.ceramic_analysis import bench_ceramic_analysis
from quant_fund.models.geoarchaeology import bench_geoarchaeology
from quant_fund.models.lithic_analysis import bench_lithic_analysis
from quant_fund.models.paleoethnobotany import bench_paleoethnobotany
from quant_fund.models.zooarchaeology import bench_zooarchaeology

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


def bench_geoarchaeology_family(seed: int = _SEED + 51000) -> dict[str, float]:
    return _finite_blob(bench_geoarchaeology(seed))


def bench_zooarchaeology_family(seed: int = _SEED + 51001) -> dict[str, float]:
    return _finite_blob(bench_zooarchaeology(seed))


def bench_paleoethnobotany_family(seed: int = _SEED + 51002) -> dict[str, float]:
    return _finite_blob(bench_paleoethnobotany(seed))


def bench_ceramic_analysis_family(seed: int = _SEED + 51003) -> dict[str, float]:
    return _finite_blob(bench_ceramic_analysis(seed))


def bench_lithic_analysis_family(seed: int = _SEED + 51004) -> dict[str, float]:
    return _finite_blob(bench_lithic_analysis(seed))


def bench_archaeogenetics_family(seed: int = _SEED + 51005) -> dict[str, float]:
    return _finite_blob(bench_archaeogenetics(seed))
