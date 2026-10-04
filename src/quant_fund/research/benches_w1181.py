"""Wave-1181 performing-arts canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.acting_studies import bench_acting_studies
from quant_fund.models.directing_studies import bench_directing_studies
from quant_fund.models.performing_arts_2 import bench_performing_arts_2
from quant_fund.models.playwriting import bench_playwriting
from quant_fund.models.scenography import bench_scenography
from quant_fund.models.theater_arts import bench_theater_arts

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


def bench_performing_arts_2_family(seed: int = _SEED + 56900) -> dict[str, float]:
    return _finite_blob(bench_performing_arts_2(seed))


def bench_theater_arts_family(seed: int = _SEED + 56901) -> dict[str, float]:
    return _finite_blob(bench_theater_arts(seed))


def bench_acting_studies_family(seed: int = _SEED + 56902) -> dict[str, float]:
    return _finite_blob(bench_acting_studies(seed))


def bench_directing_studies_family(seed: int = _SEED + 56903) -> dict[str, float]:
    return _finite_blob(bench_directing_studies(seed))


def bench_playwriting_family(seed: int = _SEED + 56904) -> dict[str, float]:
    return _finite_blob(bench_playwriting(seed))


def bench_scenography_family(seed: int = _SEED + 56905) -> dict[str, float]:
    return _finite_blob(bench_scenography(seed))
