"""Wave-1184 film-production canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.animation_studies import bench_animation_studies
from quant_fund.models.cinematography_studies import bench_cinematography_studies
from quant_fund.models.documentary_production import bench_documentary_production
from quant_fund.models.film_editing import bench_film_editing
from quant_fund.models.film_production import bench_film_production
from quant_fund.models.sound_design import bench_sound_design

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


def bench_film_production_family(seed: int = _SEED + 57200) -> dict[str, float]:
    return _finite_blob(bench_film_production(seed))


def bench_cinematography_studies_family(seed: int = _SEED + 57201) -> dict[str, float]:
    return _finite_blob(bench_cinematography_studies(seed))


def bench_film_editing_family(seed: int = _SEED + 57202) -> dict[str, float]:
    return _finite_blob(bench_film_editing(seed))


def bench_sound_design_family(seed: int = _SEED + 57203) -> dict[str, float]:
    return _finite_blob(bench_sound_design(seed))


def bench_documentary_production_family(seed: int = _SEED + 57204) -> dict[str, float]:
    return _finite_blob(bench_documentary_production(seed))


def bench_animation_studies_family(seed: int = _SEED + 57205) -> dict[str, float]:
    return _finite_blob(bench_animation_studies(seed))
