"""Wave-1169 social-policy canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.disability_studies_2 import bench_disability_studies_2
from quant_fund.models.ethnic_studies_2 import bench_ethnic_studies_2
from quant_fund.models.gender_studies_2 import bench_gender_studies_2
from quant_fund.models.public_policy_2 import bench_public_policy_2
from quant_fund.models.social_work_2 import bench_social_work_2
from quant_fund.models.urban_studies_2 import bench_urban_studies_2

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


def bench_social_work_2_family(seed: int = _SEED + 55700) -> dict[str, float]:
    return _finite_blob(bench_social_work_2(seed))


def bench_public_policy_2_family(seed: int = _SEED + 55701) -> dict[str, float]:
    return _finite_blob(bench_public_policy_2(seed))


def bench_urban_studies_2_family(seed: int = _SEED + 55702) -> dict[str, float]:
    return _finite_blob(bench_urban_studies_2(seed))


def bench_gender_studies_2_family(seed: int = _SEED + 55703) -> dict[str, float]:
    return _finite_blob(bench_gender_studies_2(seed))


def bench_ethnic_studies_2_family(seed: int = _SEED + 55704) -> dict[str, float]:
    return _finite_blob(bench_ethnic_studies_2(seed))


def bench_disability_studies_2_family(seed: int = _SEED + 55705) -> dict[str, float]:
    return _finite_blob(bench_disability_studies_2(seed))
