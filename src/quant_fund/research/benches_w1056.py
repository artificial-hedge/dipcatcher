"""Wave-1056 political-science canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.comparative_politics import bench_comparative_politics
from quant_fund.models.electoral_systems import bench_electoral_systems
from quant_fund.models.international_relations import bench_international_relations
from quant_fund.models.political_economy import bench_political_economy
from quant_fund.models.political_theory import bench_political_theory
from quant_fund.models.public_administration import bench_public_administration

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


def bench_comparative_politics_family(seed: int = _SEED + 44400) -> dict[str, float]:
    return _finite_blob(bench_comparative_politics(seed))


def bench_international_relations_family(seed: int = _SEED + 44401) -> dict[str, float]:
    return _finite_blob(bench_international_relations(seed))


def bench_political_theory_family(seed: int = _SEED + 44402) -> dict[str, float]:
    return _finite_blob(bench_political_theory(seed))


def bench_public_administration_family(seed: int = _SEED + 44403) -> dict[str, float]:
    return _finite_blob(bench_public_administration(seed))


def bench_political_economy_family(seed: int = _SEED + 44404) -> dict[str, float]:
    return _finite_blob(bench_political_economy(seed))


def bench_electoral_systems_family(seed: int = _SEED + 44405) -> dict[str, float]:
    return _finite_blob(bench_electoral_systems(seed))
