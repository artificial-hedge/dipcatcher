"""Wave-1207 psychotherapy canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.art_therapy import bench_art_therapy
from quant_fund.models.behavioral_therapy_cognitive import bench_behavioral_therapy_cognitive
from quant_fund.models.music_therapy import bench_music_therapy
from quant_fund.models.play_therapy import bench_play_therapy
from quant_fund.models.psychoanalysis_studies import bench_psychoanalysis_studies
from quant_fund.models.psychotherapy_studies import bench_psychotherapy_studies

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


def bench_psychoanalysis_studies_family(seed: int = _SEED + 59500) -> dict[str, float]:
    return _finite_blob(bench_psychoanalysis_studies(seed))


def bench_psychotherapy_studies_family(seed: int = _SEED + 59501) -> dict[str, float]:
    return _finite_blob(bench_psychotherapy_studies(seed))


def bench_behavioral_therapy_cognitive_family(seed: int = _SEED + 59502) -> dict[str, float]:
    return _finite_blob(bench_behavioral_therapy_cognitive(seed))


def bench_art_therapy_family(seed: int = _SEED + 59503) -> dict[str, float]:
    return _finite_blob(bench_art_therapy(seed))


def bench_music_therapy_family(seed: int = _SEED + 59504) -> dict[str, float]:
    return _finite_blob(bench_music_therapy(seed))


def bench_play_therapy_family(seed: int = _SEED + 59505) -> dict[str, float]:
    return _finite_blob(bench_play_therapy(seed))
