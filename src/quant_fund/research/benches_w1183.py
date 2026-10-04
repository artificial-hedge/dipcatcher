"""Wave-1183 music canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.composition_studies import bench_composition_studies
from quant_fund.models.ethnomusicology_2 import bench_ethnomusicology_2
from quant_fund.models.music_cognition_2 import bench_music_cognition_2
from quant_fund.models.music_theory_2 import bench_music_theory_2
from quant_fund.models.musicology_2 import bench_musicology_2
from quant_fund.models.organology_2 import bench_organology_2

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


def bench_music_theory_2_family(seed: int = _SEED + 57100) -> dict[str, float]:
    return _finite_blob(bench_music_theory_2(seed))


def bench_musicology_2_family(seed: int = _SEED + 57101) -> dict[str, float]:
    return _finite_blob(bench_musicology_2(seed))


def bench_ethnomusicology_2_family(seed: int = _SEED + 57102) -> dict[str, float]:
    return _finite_blob(bench_ethnomusicology_2(seed))


def bench_music_cognition_2_family(seed: int = _SEED + 57103) -> dict[str, float]:
    return _finite_blob(bench_music_cognition_2(seed))


def bench_organology_2_family(seed: int = _SEED + 57104) -> dict[str, float]:
    return _finite_blob(bench_organology_2(seed))


def bench_composition_studies_family(seed: int = _SEED + 57105) -> dict[str, float]:
    return _finite_blob(bench_composition_studies(seed))
