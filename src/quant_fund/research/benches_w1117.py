"""Wave-1117 sociology-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.historical_sociology import bench_historical_sociology
from quant_fund.models.legal_sociology import bench_legal_sociology
from quant_fund.models.mathematical_sociology import bench_mathematical_sociology
from quant_fund.models.military_sociology import bench_military_sociology
from quant_fund.models.science_studies import bench_science_studies
from quant_fund.models.sociology_of_knowledge import bench_sociology_of_knowledge

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


def bench_mathematical_sociology_family(seed: int = _SEED + 50500) -> dict[str, float]:
    return _finite_blob(bench_mathematical_sociology(seed))


def bench_historical_sociology_family(seed: int = _SEED + 50501) -> dict[str, float]:
    return _finite_blob(bench_historical_sociology(seed))


def bench_science_studies_family(seed: int = _SEED + 50502) -> dict[str, float]:
    return _finite_blob(bench_science_studies(seed))


def bench_sociology_of_knowledge_family(seed: int = _SEED + 50503) -> dict[str, float]:
    return _finite_blob(bench_sociology_of_knowledge(seed))


def bench_military_sociology_family(seed: int = _SEED + 50504) -> dict[str, float]:
    return _finite_blob(bench_military_sociology(seed))


def bench_legal_sociology_family(seed: int = _SEED + 50505) -> dict[str, float]:
    return _finite_blob(bench_legal_sociology(seed))
