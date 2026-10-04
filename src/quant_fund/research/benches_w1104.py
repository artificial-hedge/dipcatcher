"""Wave-1104 political-science-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.american_politics import bench_american_politics
from quant_fund.models.policy_analysis import bench_policy_analysis
from quant_fund.models.political_behavior import bench_political_behavior
from quant_fund.models.political_methodology import bench_political_methodology
from quant_fund.models.public_law import bench_public_law
from quant_fund.models.security_studies import bench_security_studies

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


def bench_american_politics_family(seed: int = _SEED + 49200) -> dict[str, float]:
    return _finite_blob(bench_american_politics(seed))


def bench_political_behavior_family(seed: int = _SEED + 49201) -> dict[str, float]:
    return _finite_blob(bench_political_behavior(seed))


def bench_public_law_family(seed: int = _SEED + 49202) -> dict[str, float]:
    return _finite_blob(bench_public_law(seed))


def bench_political_methodology_family(seed: int = _SEED + 49203) -> dict[str, float]:
    return _finite_blob(bench_political_methodology(seed))


def bench_security_studies_family(seed: int = _SEED + 49204) -> dict[str, float]:
    return _finite_blob(bench_security_studies(seed))


def bench_policy_analysis_family(seed: int = _SEED + 49205) -> dict[str, float]:
    return _finite_blob(bench_policy_analysis(seed))
