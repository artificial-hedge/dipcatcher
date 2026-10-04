"""Wave-1145 life-science canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bioinformatics_4 import bench_bioinformatics_4
from quant_fund.models.epidemiology_3 import bench_epidemiology_3
from quant_fund.models.genomicsciences import bench_genomicsciences
from quant_fund.models.proteomics import bench_proteomics
from quant_fund.models.synthetic_biology import bench_synthetic_biology
from quant_fund.models.systems_biology_2 import bench_systems_biology_2

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


def bench_genomicsciences_family(seed: int = _SEED + 53300) -> dict[str, float]:
    return _finite_blob(bench_genomicsciences(seed))


def bench_proteomics_family(seed: int = _SEED + 53301) -> dict[str, float]:
    return _finite_blob(bench_proteomics(seed))


def bench_bioinformatics_4_family(seed: int = _SEED + 53302) -> dict[str, float]:
    return _finite_blob(bench_bioinformatics_4(seed))


def bench_systems_biology_2_family(seed: int = _SEED + 53303) -> dict[str, float]:
    return _finite_blob(bench_systems_biology_2(seed))


def bench_synthetic_biology_family(seed: int = _SEED + 53304) -> dict[str, float]:
    return _finite_blob(bench_synthetic_biology(seed))


def bench_epidemiology_3_family(seed: int = _SEED + 53305) -> dict[str, float]:
    return _finite_blob(bench_epidemiology_3(seed))
