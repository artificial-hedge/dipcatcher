"""Wave-1086 near-eastern/ancient canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.assyriology import bench_assyriology
from quant_fund.models.egyptology import bench_egyptology
from quant_fund.models.indology import bench_indology
from quant_fund.models.iranian_studies import bench_iranian_studies
from quant_fund.models.ottoman_studies import bench_ottoman_studies
from quant_fund.models.sinology import bench_sinology

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


def bench_assyriology_family(seed: int = _SEED + 47400) -> dict[str, float]:
    return _finite_blob(bench_assyriology(seed))


def bench_egyptology_family(seed: int = _SEED + 47401) -> dict[str, float]:
    return _finite_blob(bench_egyptology(seed))


def bench_sinology_family(seed: int = _SEED + 47402) -> dict[str, float]:
    return _finite_blob(bench_sinology(seed))


def bench_indology_family(seed: int = _SEED + 47403) -> dict[str, float]:
    return _finite_blob(bench_indology(seed))


def bench_iranian_studies_family(seed: int = _SEED + 47404) -> dict[str, float]:
    return _finite_blob(bench_iranian_studies(seed))


def bench_ottoman_studies_family(seed: int = _SEED + 47405) -> dict[str, float]:
    return _finite_blob(bench_ottoman_studies(seed))
