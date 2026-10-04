"""Wave-1123 linguistics-5 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.contact_linguistics import bench_contact_linguistics
from quant_fund.models.descriptive_linguistics import bench_descriptive_linguistics
from quant_fund.models.dialectometry import bench_dialectometry
from quant_fund.models.etymology import bench_etymology
from quant_fund.models.lexicography import bench_lexicography
from quant_fund.models.philological_studies import bench_philological_studies

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


def bench_contact_linguistics_family(seed: int = _SEED + 51100) -> dict[str, float]:
    return _finite_blob(bench_contact_linguistics(seed))


def bench_descriptive_linguistics_family(seed: int = _SEED + 51101) -> dict[str, float]:
    return _finite_blob(bench_descriptive_linguistics(seed))


def bench_philological_studies_family(seed: int = _SEED + 51102) -> dict[str, float]:
    return _finite_blob(bench_philological_studies(seed))


def bench_etymology_family(seed: int = _SEED + 51103) -> dict[str, float]:
    return _finite_blob(bench_etymology(seed))


def bench_dialectometry_family(seed: int = _SEED + 51104) -> dict[str, float]:
    return _finite_blob(bench_dialectometry(seed))


def bench_lexicography_family(seed: int = _SEED + 51105) -> dict[str, float]:
    return _finite_blob(bench_lexicography(seed))
