"""Wave-1158 social-sciences canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.anthropology_6 import bench_anthropology_6
from quant_fund.models.economics_6 import bench_economics_6
from quant_fund.models.linguistics_7 import bench_linguistics_7
from quant_fund.models.political_science_3 import bench_political_science_3
from quant_fund.models.psychology_5 import bench_psychology_5
from quant_fund.models.sociology_6 import bench_sociology_6

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


def bench_sociology_6_family(seed: int = _SEED + 54600) -> dict[str, float]:
    return _finite_blob(bench_sociology_6(seed))


def bench_economics_6_family(seed: int = _SEED + 54601) -> dict[str, float]:
    return _finite_blob(bench_economics_6(seed))


def bench_political_science_3_family(seed: int = _SEED + 54602) -> dict[str, float]:
    return _finite_blob(bench_political_science_3(seed))


def bench_psychology_5_family(seed: int = _SEED + 54603) -> dict[str, float]:
    return _finite_blob(bench_psychology_5(seed))


def bench_anthropology_6_family(seed: int = _SEED + 54604) -> dict[str, float]:
    return _finite_blob(bench_anthropology_6(seed))


def bench_linguistics_7_family(seed: int = _SEED + 54605) -> dict[str, float]:
    return _finite_blob(bench_linguistics_7(seed))
