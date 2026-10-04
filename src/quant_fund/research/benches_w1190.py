"""Wave-1190 visual-design canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.graphic_design import bench_graphic_design
from quant_fund.models.motion_graphics import bench_motion_graphics
from quant_fund.models.photography_studies import bench_photography_studies
from quant_fund.models.print_media import bench_print_media
from quant_fund.models.typography_studies import bench_typography_studies
from quant_fund.models.web_design import bench_web_design

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


def bench_graphic_design_family(seed: int = _SEED + 57800) -> dict[str, float]:
    return _finite_blob(bench_graphic_design(seed))


def bench_typography_studies_family(seed: int = _SEED + 57801) -> dict[str, float]:
    return _finite_blob(bench_typography_studies(seed))


def bench_photography_studies_family(seed: int = _SEED + 57802) -> dict[str, float]:
    return _finite_blob(bench_photography_studies(seed))


def bench_print_media_family(seed: int = _SEED + 57803) -> dict[str, float]:
    return _finite_blob(bench_print_media(seed))


def bench_web_design_family(seed: int = _SEED + 57804) -> dict[str, float]:
    return _finite_blob(bench_web_design(seed))


def bench_motion_graphics_family(seed: int = _SEED + 57805) -> dict[str, float]:
    return _finite_blob(bench_motion_graphics(seed))
