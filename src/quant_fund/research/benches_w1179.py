"""Wave-1179 security canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.conflict_studies import bench_conflict_studies
from quant_fund.models.intelligence_analysis import bench_intelligence_analysis
from quant_fund.models.military_history_2 import bench_military_history_2
from quant_fund.models.peace_research import bench_peace_research
from quant_fund.models.strategic_analysis import bench_strategic_analysis
from quant_fund.models.war_studies import bench_war_studies

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


def bench_war_studies_family(seed: int = _SEED + 56700) -> dict[str, float]:
    return _finite_blob(bench_war_studies(seed))


def bench_strategic_analysis_family(seed: int = _SEED + 56701) -> dict[str, float]:
    return _finite_blob(bench_strategic_analysis(seed))


def bench_intelligence_analysis_family(seed: int = _SEED + 56702) -> dict[str, float]:
    return _finite_blob(bench_intelligence_analysis(seed))


def bench_peace_research_family(seed: int = _SEED + 56703) -> dict[str, float]:
    return _finite_blob(bench_peace_research(seed))


def bench_conflict_studies_family(seed: int = _SEED + 56704) -> dict[str, float]:
    return _finite_blob(bench_conflict_studies(seed))


def bench_military_history_2_family(seed: int = _SEED + 56705) -> dict[str, float]:
    return _finite_blob(bench_military_history_2(seed))
