"""Wave-1093 law-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.canon_law import bench_canon_law
from quant_fund.models.civil_law import bench_civil_law
from quant_fund.models.common_law import bench_common_law
from quant_fund.models.maritime_law import bench_maritime_law
from quant_fund.models.procedural_law import bench_procedural_law
from quant_fund.models.property_law import bench_property_law

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


def bench_civil_law_family(seed: int = _SEED + 48100) -> dict[str, float]:
    return _finite_blob(bench_civil_law(seed))


def bench_common_law_family(seed: int = _SEED + 48101) -> dict[str, float]:
    return _finite_blob(bench_common_law(seed))


def bench_canon_law_family(seed: int = _SEED + 48102) -> dict[str, float]:
    return _finite_blob(bench_canon_law(seed))


def bench_maritime_law_family(seed: int = _SEED + 48103) -> dict[str, float]:
    return _finite_blob(bench_maritime_law(seed))


def bench_property_law_family(seed: int = _SEED + 48104) -> dict[str, float]:
    return _finite_blob(bench_property_law(seed))


def bench_procedural_law_family(seed: int = _SEED + 48105) -> dict[str, float]:
    return _finite_blob(bench_procedural_law(seed))
