"""Wave-1109 linguistics-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.anthropological_linguistics import bench_anthropological_linguistics
from quant_fund.models.applied_linguistics import bench_applied_linguistics
from quant_fund.models.discourse_analysis import bench_discourse_analysis
from quant_fund.models.evolutionary_linguistics import bench_evolutionary_linguistics
from quant_fund.models.forensic_linguistics import bench_forensic_linguistics
from quant_fund.models.neurolinguistics import bench_neurolinguistics

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


def bench_applied_linguistics_family(seed: int = _SEED + 49700) -> dict[str, float]:
    return _finite_blob(bench_applied_linguistics(seed))


def bench_anthropological_linguistics_family(seed: int = _SEED + 49701) -> dict[str, float]:
    return _finite_blob(bench_anthropological_linguistics(seed))


def bench_neurolinguistics_family(seed: int = _SEED + 49702) -> dict[str, float]:
    return _finite_blob(bench_neurolinguistics(seed))


def bench_evolutionary_linguistics_family(seed: int = _SEED + 49703) -> dict[str, float]:
    return _finite_blob(bench_evolutionary_linguistics(seed))


def bench_forensic_linguistics_family(seed: int = _SEED + 49704) -> dict[str, float]:
    return _finite_blob(bench_forensic_linguistics(seed))


def bench_discourse_analysis_family(seed: int = _SEED + 49705) -> dict[str, float]:
    return _finite_blob(bench_discourse_analysis(seed))
