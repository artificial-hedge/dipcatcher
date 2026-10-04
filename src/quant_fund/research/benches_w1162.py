"""Wave-1162 governance canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.criminology_2 import bench_criminology_2
from quant_fund.models.international_relations_2 import bench_international_relations_2
from quant_fund.models.law_5 import bench_law_5
from quant_fund.models.military_science_2 import bench_military_science_2
from quant_fund.models.political_science_4 import bench_political_science_4
from quant_fund.models.public_administration_2 import bench_public_administration_2

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


def bench_law_5_family(seed: int = _SEED + 55000) -> dict[str, float]:
    return _finite_blob(bench_law_5(seed))


def bench_political_science_4_family(seed: int = _SEED + 55001) -> dict[str, float]:
    return _finite_blob(bench_political_science_4(seed))


def bench_public_administration_2_family(seed: int = _SEED + 55002) -> dict[str, float]:
    return _finite_blob(bench_public_administration_2(seed))


def bench_international_relations_2_family(seed: int = _SEED + 55003) -> dict[str, float]:
    return _finite_blob(bench_international_relations_2(seed))


def bench_criminology_2_family(seed: int = _SEED + 55004) -> dict[str, float]:
    return _finite_blob(bench_criminology_2(seed))


def bench_military_science_2_family(seed: int = _SEED + 55005) -> dict[str, float]:
    return _finite_blob(bench_military_science_2(seed))
