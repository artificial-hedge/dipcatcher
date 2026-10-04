"""Wave-1084 comparative-literature canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.comparative_literature import bench_comparative_literature
from quant_fund.models.critical_theory import bench_critical_theory
from quant_fund.models.literary_theory import bench_literary_theory
from quant_fund.models.postcolonial_studies import bench_postcolonial_studies
from quant_fund.models.translation_studies import bench_translation_studies
from quant_fund.models.world_literature import bench_world_literature

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


def bench_comparative_literature_family(seed: int = _SEED + 47200) -> dict[str, float]:
    return _finite_blob(bench_comparative_literature(seed))


def bench_literary_theory_family(seed: int = _SEED + 47201) -> dict[str, float]:
    return _finite_blob(bench_literary_theory(seed))


def bench_postcolonial_studies_family(seed: int = _SEED + 47202) -> dict[str, float]:
    return _finite_blob(bench_postcolonial_studies(seed))


def bench_world_literature_family(seed: int = _SEED + 47203) -> dict[str, float]:
    return _finite_blob(bench_world_literature(seed))


def bench_translation_studies_family(seed: int = _SEED + 47204) -> dict[str, float]:
    return _finite_blob(bench_translation_studies(seed))


def bench_critical_theory_family(seed: int = _SEED + 47205) -> dict[str, float]:
    return _finite_blob(bench_critical_theory(seed))
