"""Wave-1080 medieval studies canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.byzantine_studies import bench_byzantine_studies
from quant_fund.models.codicology import bench_codicology
from quant_fund.models.hagiography import bench_hagiography
from quant_fund.models.medieval_studies import bench_medieval_studies
from quant_fund.models.numismatics import bench_numismatics
from quant_fund.models.paleography import bench_paleography

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


def bench_medieval_studies_family(seed: int = _SEED + 46800) -> dict[str, float]:
    return _finite_blob(bench_medieval_studies(seed))


def bench_paleography_family(seed: int = _SEED + 46801) -> dict[str, float]:
    return _finite_blob(bench_paleography(seed))


def bench_codicology_family(seed: int = _SEED + 46802) -> dict[str, float]:
    return _finite_blob(bench_codicology(seed))


def bench_hagiography_family(seed: int = _SEED + 46803) -> dict[str, float]:
    return _finite_blob(bench_hagiography(seed))


def bench_byzantine_studies_family(seed: int = _SEED + 46804) -> dict[str, float]:
    return _finite_blob(bench_byzantine_studies(seed))


def bench_numismatics_family(seed: int = _SEED + 46805) -> dict[str, float]:
    return _finite_blob(bench_numismatics(seed))
