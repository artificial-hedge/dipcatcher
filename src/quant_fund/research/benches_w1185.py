"""Wave-1185 game canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.esports_studies import bench_esports_studies
from quant_fund.models.game_design import bench_game_design
from quant_fund.models.game_development import bench_game_development
from quant_fund.models.game_studies import bench_game_studies
from quant_fund.models.interactive_media import bench_interactive_media
from quant_fund.models.ludology import bench_ludology

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


def bench_game_design_family(seed: int = _SEED + 57300) -> dict[str, float]:
    return _finite_blob(bench_game_design(seed))


def bench_esports_studies_family(seed: int = _SEED + 57301) -> dict[str, float]:
    return _finite_blob(bench_esports_studies(seed))


def bench_interactive_media_family(seed: int = _SEED + 57302) -> dict[str, float]:
    return _finite_blob(bench_interactive_media(seed))


def bench_game_studies_family(seed: int = _SEED + 57303) -> dict[str, float]:
    return _finite_blob(bench_game_studies(seed))


def bench_ludology_family(seed: int = _SEED + 57304) -> dict[str, float]:
    return _finite_blob(bench_ludology(seed))


def bench_game_development_family(seed: int = _SEED + 57305) -> dict[str, float]:
    return _finite_blob(bench_game_development(seed))
