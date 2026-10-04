"""Wave-1064 social-work/policy canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.disability_studies import bench_disability_studies
from quant_fund.models.ethnic_studies import bench_ethnic_studies
from quant_fund.models.gender_studies import bench_gender_studies
from quant_fund.models.public_policy import bench_public_policy
from quant_fund.models.social_work import bench_social_work
from quant_fund.models.urban_studies import bench_urban_studies

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


def bench_social_work_family(seed: int = _SEED + 45200) -> dict[str, float]:
    return _finite_blob(bench_social_work(seed))


def bench_public_policy_family(seed: int = _SEED + 45201) -> dict[str, float]:
    return _finite_blob(bench_public_policy(seed))


def bench_urban_studies_family(seed: int = _SEED + 45202) -> dict[str, float]:
    return _finite_blob(bench_urban_studies(seed))


def bench_gender_studies_family(seed: int = _SEED + 45203) -> dict[str, float]:
    return _finite_blob(bench_gender_studies(seed))


def bench_ethnic_studies_family(seed: int = _SEED + 45204) -> dict[str, float]:
    return _finite_blob(bench_ethnic_studies(seed))


def bench_disability_studies_family(seed: int = _SEED + 45205) -> dict[str, float]:
    return _finite_blob(bench_disability_studies(seed))
