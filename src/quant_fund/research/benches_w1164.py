"""Wave-1164 communication canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.communication_studies_2 import bench_communication_studies_2
from quant_fund.models.education_5 import bench_education_5
from quant_fund.models.information_science_2 import bench_information_science_2
from quant_fund.models.journalism_2 import bench_journalism_2
from quant_fund.models.library_science_2 import bench_library_science_2
from quant_fund.models.media_studies_2 import bench_media_studies_2

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


def bench_education_5_family(seed: int = _SEED + 55200) -> dict[str, float]:
    return _finite_blob(bench_education_5(seed))


def bench_communication_studies_2_family(seed: int = _SEED + 55201) -> dict[str, float]:
    return _finite_blob(bench_communication_studies_2(seed))


def bench_media_studies_2_family(seed: int = _SEED + 55202) -> dict[str, float]:
    return _finite_blob(bench_media_studies_2(seed))


def bench_journalism_2_family(seed: int = _SEED + 55203) -> dict[str, float]:
    return _finite_blob(bench_journalism_2(seed))


def bench_library_science_2_family(seed: int = _SEED + 55204) -> dict[str, float]:
    return _finite_blob(bench_library_science_2(seed))


def bench_information_science_2_family(seed: int = _SEED + 55205) -> dict[str, float]:
    return _finite_blob(bench_information_science_2(seed))
