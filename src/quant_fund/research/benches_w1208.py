"""Wave-1208 therapy-modalities canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.child_adolescent_therapy import bench_child_adolescent_therapy
from quant_fund.models.couples_therapy import bench_couples_therapy
from quant_fund.models.family_therapy import bench_family_therapy
from quant_fund.models.group_therapy import bench_group_therapy
from quant_fund.models.marriage_family_therapy import bench_marriage_family_therapy
from quant_fund.models.trauma_therapy import bench_trauma_therapy

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


def bench_marriage_family_therapy_family(seed: int = _SEED + 59600) -> dict[str, float]:
    return _finite_blob(bench_marriage_family_therapy(seed))


def bench_group_therapy_family(seed: int = _SEED + 59601) -> dict[str, float]:
    return _finite_blob(bench_group_therapy(seed))


def bench_couples_therapy_family(seed: int = _SEED + 59602) -> dict[str, float]:
    return _finite_blob(bench_couples_therapy(seed))


def bench_family_therapy_family(seed: int = _SEED + 59603) -> dict[str, float]:
    return _finite_blob(bench_family_therapy(seed))


def bench_child_adolescent_therapy_family(seed: int = _SEED + 59604) -> dict[str, float]:
    return _finite_blob(bench_child_adolescent_therapy(seed))


def bench_trauma_therapy_family(seed: int = _SEED + 59605) -> dict[str, float]:
    return _finite_blob(bench_trauma_therapy(seed))
