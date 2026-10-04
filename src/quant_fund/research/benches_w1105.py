"""Wave-1105 behavioral-econ-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bounded_rationality import bench_bounded_rationality
from quant_fund.models.experimental_economics import bench_experimental_economics
from quant_fund.models.financial_behavior import bench_financial_behavior
from quant_fund.models.neuroeconomics import bench_neuroeconomics
from quant_fund.models.nudge_theory import bench_nudge_theory
from quant_fund.models.prospect_theory import bench_prospect_theory

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


def bench_prospect_theory_family(seed: int = _SEED + 49300) -> dict[str, float]:
    return _finite_blob(bench_prospect_theory(seed))


def bench_bounded_rationality_family(seed: int = _SEED + 49301) -> dict[str, float]:
    return _finite_blob(bench_bounded_rationality(seed))


def bench_nudge_theory_family(seed: int = _SEED + 49302) -> dict[str, float]:
    return _finite_blob(bench_nudge_theory(seed))


def bench_neuroeconomics_family(seed: int = _SEED + 49303) -> dict[str, float]:
    return _finite_blob(bench_neuroeconomics(seed))


def bench_experimental_economics_family(seed: int = _SEED + 49304) -> dict[str, float]:
    return _finite_blob(bench_experimental_economics(seed))


def bench_financial_behavior_family(seed: int = _SEED + 49305) -> dict[str, float]:
    return _finite_blob(bench_financial_behavior(seed))
