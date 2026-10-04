"""Wave-1071 musicology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.ethnomusicology import bench_ethnomusicology
from quant_fund.models.music_cognition import bench_music_cognition
from quant_fund.models.music_history import bench_music_history
from quant_fund.models.music_theory import bench_music_theory
from quant_fund.models.musicology import bench_musicology
from quant_fund.models.organology import bench_organology

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


def bench_musicology_family(seed: int = _SEED + 45900) -> dict[str, float]:
    return _finite_blob(bench_musicology(seed))


def bench_ethnomusicology_family(seed: int = _SEED + 45901) -> dict[str, float]:
    return _finite_blob(bench_ethnomusicology(seed))


def bench_music_theory_family(seed: int = _SEED + 45902) -> dict[str, float]:
    return _finite_blob(bench_music_theory(seed))


def bench_music_cognition_family(seed: int = _SEED + 45903) -> dict[str, float]:
    return _finite_blob(bench_music_cognition(seed))


def bench_organology_family(seed: int = _SEED + 45904) -> dict[str, float]:
    return _finite_blob(bench_organology(seed))


def bench_music_history_family(seed: int = _SEED + 45905) -> dict[str, float]:
    return _finite_blob(bench_music_history(seed))
