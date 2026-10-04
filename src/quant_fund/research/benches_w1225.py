"""Wave-1225 surgery canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.colorectal_surgery import bench_colorectal_surgery
from quant_fund.models.general_surgery_studies import bench_general_surgery_studies
from quant_fund.models.hepatobiliary_surgery import bench_hepatobiliary_surgery
from quant_fund.models.minimally_invasive_surgery import bench_minimally_invasive_surgery
from quant_fund.models.surgical_oncology_studies import bench_surgical_oncology_studies
from quant_fund.models.trauma_surgery import bench_trauma_surgery

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


def bench_general_surgery_studies_family(seed: int = _SEED + 61300) -> dict[str, float]:
    return _finite_blob(bench_general_surgery_studies(seed))


def bench_trauma_surgery_family(seed: int = _SEED + 61301) -> dict[str, float]:
    return _finite_blob(bench_trauma_surgery(seed))


def bench_colorectal_surgery_family(seed: int = _SEED + 61302) -> dict[str, float]:
    return _finite_blob(bench_colorectal_surgery(seed))


def bench_hepatobiliary_surgery_family(seed: int = _SEED + 61303) -> dict[str, float]:
    return _finite_blob(bench_hepatobiliary_surgery(seed))


def bench_surgical_oncology_studies_family(seed: int = _SEED + 61304) -> dict[str, float]:
    return _finite_blob(bench_surgical_oncology_studies(seed))


def bench_minimally_invasive_surgery_family(seed: int = _SEED + 61305) -> dict[str, float]:
    return _finite_blob(bench_minimally_invasive_surgery(seed))
