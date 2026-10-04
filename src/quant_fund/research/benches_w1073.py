"""Wave-1073 theology-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.biblical_exegesis import bench_biblical_exegesis
from quant_fund.models.church_history import bench_church_history
from quant_fund.models.liturgical_studies import bench_liturgical_studies
from quant_fund.models.missiology import bench_missiology
from quant_fund.models.pastoral_theology import bench_pastoral_theology
from quant_fund.models.systematic_theology import bench_systematic_theology

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


def bench_systematic_theology_family(seed: int = _SEED + 46100) -> dict[str, float]:
    return _finite_blob(bench_systematic_theology(seed))


def bench_biblical_exegesis_family(seed: int = _SEED + 46101) -> dict[str, float]:
    return _finite_blob(bench_biblical_exegesis(seed))


def bench_church_history_family(seed: int = _SEED + 46102) -> dict[str, float]:
    return _finite_blob(bench_church_history(seed))


def bench_pastoral_theology_family(seed: int = _SEED + 46103) -> dict[str, float]:
    return _finite_blob(bench_pastoral_theology(seed))


def bench_liturgical_studies_family(seed: int = _SEED + 46104) -> dict[str, float]:
    return _finite_blob(bench_liturgical_studies(seed))


def bench_missiology_family(seed: int = _SEED + 46105) -> dict[str, float]:
    return _finite_blob(bench_missiology(seed))
