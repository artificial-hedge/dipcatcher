"""Wave-1062 religious-studies canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.biblical_studies import bench_biblical_studies
from quant_fund.models.buddhist_studies import bench_buddhist_studies
from quant_fund.models.comparative_religion import bench_comparative_religion
from quant_fund.models.islamic_studies import bench_islamic_studies
from quant_fund.models.religious_ethics import bench_religious_ethics
from quant_fund.models.theology import bench_theology

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


def bench_theology_family(seed: int = _SEED + 45000) -> dict[str, float]:
    return _finite_blob(bench_theology(seed))


def bench_comparative_religion_family(seed: int = _SEED + 45001) -> dict[str, float]:
    return _finite_blob(bench_comparative_religion(seed))


def bench_biblical_studies_family(seed: int = _SEED + 45002) -> dict[str, float]:
    return _finite_blob(bench_biblical_studies(seed))


def bench_islamic_studies_family(seed: int = _SEED + 45003) -> dict[str, float]:
    return _finite_blob(bench_islamic_studies(seed))


def bench_buddhist_studies_family(seed: int = _SEED + 45004) -> dict[str, float]:
    return _finite_blob(bench_buddhist_studies(seed))


def bench_religious_ethics_family(seed: int = _SEED + 45005) -> dict[str, float]:
    return _finite_blob(bench_religious_ethics(seed))
