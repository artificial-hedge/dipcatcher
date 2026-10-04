"""Wave-1120 anthropology-4 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.applied_anthropology import bench_applied_anthropology
from quant_fund.models.digital_anthropology import bench_digital_anthropology
from quant_fund.models.environmental_anthropology import bench_environmental_anthropology
from quant_fund.models.forensic_anthropology import bench_forensic_anthropology
from quant_fund.models.psychological_anthropology import bench_psychological_anthropology
from quant_fund.models.visual_anthropology import bench_visual_anthropology

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


def bench_visual_anthropology_family(seed: int = _SEED + 50800) -> dict[str, float]:
    return _finite_blob(bench_visual_anthropology(seed))


def bench_applied_anthropology_family(seed: int = _SEED + 50801) -> dict[str, float]:
    return _finite_blob(bench_applied_anthropology(seed))


def bench_forensic_anthropology_family(seed: int = _SEED + 50802) -> dict[str, float]:
    return _finite_blob(bench_forensic_anthropology(seed))


def bench_digital_anthropology_family(seed: int = _SEED + 50803) -> dict[str, float]:
    return _finite_blob(bench_digital_anthropology(seed))


def bench_environmental_anthropology_family(seed: int = _SEED + 50804) -> dict[str, float]:
    return _finite_blob(bench_environmental_anthropology(seed))


def bench_psychological_anthropology_family(seed: int = _SEED + 50805) -> dict[str, float]:
    return _finite_blob(bench_psychological_anthropology(seed))
