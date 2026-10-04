"""Wave-1209 behavioral-health canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.addiction_medicine import bench_addiction_medicine
from quant_fund.models.community_psychiatry import bench_community_psychiatry
from quant_fund.models.consultation_liaison import bench_consultation_liaison
from quant_fund.models.eating_disorders import bench_eating_disorders
from quant_fund.models.psychosomatic_medicine import bench_psychosomatic_medicine
from quant_fund.models.sleep_disorders import bench_sleep_disorders

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


def bench_addiction_medicine_family(seed: int = _SEED + 59700) -> dict[str, float]:
    return _finite_blob(bench_addiction_medicine(seed))


def bench_eating_disorders_family(seed: int = _SEED + 59701) -> dict[str, float]:
    return _finite_blob(bench_eating_disorders(seed))


def bench_sleep_disorders_family(seed: int = _SEED + 59702) -> dict[str, float]:
    return _finite_blob(bench_sleep_disorders(seed))


def bench_psychosomatic_medicine_family(seed: int = _SEED + 59703) -> dict[str, float]:
    return _finite_blob(bench_psychosomatic_medicine(seed))


def bench_consultation_liaison_family(seed: int = _SEED + 59704) -> dict[str, float]:
    return _finite_blob(bench_consultation_liaison(seed))


def bench_community_psychiatry_family(seed: int = _SEED + 59705) -> dict[str, float]:
    return _finite_blob(bench_community_psychiatry(seed))
