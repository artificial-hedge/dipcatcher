"""Wave-1176 theology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.biblical_studies_2 import bench_biblical_studies_2
from quant_fund.models.buddhist_studies_2 import bench_buddhist_studies_2
from quant_fund.models.comparative_religion_2 import bench_comparative_religion_2
from quant_fund.models.islamic_studies_2 import bench_islamic_studies_2
from quant_fund.models.religious_studies_3 import bench_religious_studies_3
from quant_fund.models.theology_3 import bench_theology_3

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


def bench_theology_3_family(seed: int = _SEED + 56400) -> dict[str, float]:
    return _finite_blob(bench_theology_3(seed))


def bench_religious_studies_3_family(seed: int = _SEED + 56401) -> dict[str, float]:
    return _finite_blob(bench_religious_studies_3(seed))


def bench_comparative_religion_2_family(seed: int = _SEED + 56402) -> dict[str, float]:
    return _finite_blob(bench_comparative_religion_2(seed))


def bench_biblical_studies_2_family(seed: int = _SEED + 56403) -> dict[str, float]:
    return _finite_blob(bench_biblical_studies_2(seed))


def bench_islamic_studies_2_family(seed: int = _SEED + 56404) -> dict[str, float]:
    return _finite_blob(bench_islamic_studies_2(seed))


def bench_buddhist_studies_2_family(seed: int = _SEED + 56405) -> dict[str, float]:
    return _finite_blob(bench_buddhist_studies_2(seed))
