"""Wave-1222 ortho canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.hand_surgery import bench_hand_surgery
from quant_fund.models.joint_replacement import bench_joint_replacement
from quant_fund.models.musculoskeletal_medicine import bench_musculoskeletal_medicine
from quant_fund.models.orthopedics_studies import bench_orthopedics_studies
from quant_fund.models.spine_surgery import bench_spine_surgery
from quant_fund.models.sports_medicine_orthopedics import bench_sports_medicine_orthopedics

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


def bench_orthopedics_studies_family(seed: int = _SEED + 61000) -> dict[str, float]:
    return _finite_blob(bench_orthopedics_studies(seed))


def bench_sports_medicine_orthopedics_family(seed: int = _SEED + 61001) -> dict[str, float]:
    return _finite_blob(bench_sports_medicine_orthopedics(seed))


def bench_musculoskeletal_medicine_family(seed: int = _SEED + 61002) -> dict[str, float]:
    return _finite_blob(bench_musculoskeletal_medicine(seed))


def bench_spine_surgery_family(seed: int = _SEED + 61003) -> dict[str, float]:
    return _finite_blob(bench_spine_surgery(seed))


def bench_joint_replacement_family(seed: int = _SEED + 61004) -> dict[str, float]:
    return _finite_blob(bench_joint_replacement(seed))


def bench_hand_surgery_family(seed: int = _SEED + 61005) -> dict[str, float]:
    return _finite_blob(bench_hand_surgery(seed))
