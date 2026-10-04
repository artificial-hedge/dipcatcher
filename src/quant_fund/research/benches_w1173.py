"""Wave-1173 media canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.communication_3 import bench_communication_3
from quant_fund.models.digital_media_2 import bench_digital_media_2
from quant_fund.models.information_science_3 import bench_information_science_3
from quant_fund.models.journalism_3 import bench_journalism_3
from quant_fund.models.media_studies_3 import bench_media_studies_3
from quant_fund.models.rhetoric_2 import bench_rhetoric_2

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


def bench_communication_3_family(seed: int = _SEED + 56100) -> dict[str, float]:
    return _finite_blob(bench_communication_3(seed))


def bench_journalism_3_family(seed: int = _SEED + 56101) -> dict[str, float]:
    return _finite_blob(bench_journalism_3(seed))


def bench_media_studies_3_family(seed: int = _SEED + 56102) -> dict[str, float]:
    return _finite_blob(bench_media_studies_3(seed))


def bench_rhetoric_2_family(seed: int = _SEED + 56103) -> dict[str, float]:
    return _finite_blob(bench_rhetoric_2(seed))


def bench_information_science_3_family(seed: int = _SEED + 56104) -> dict[str, float]:
    return _finite_blob(bench_information_science_3(seed))


def bench_digital_media_2_family(seed: int = _SEED + 56105) -> dict[str, float]:
    return _finite_blob(bench_digital_media_2(seed))
