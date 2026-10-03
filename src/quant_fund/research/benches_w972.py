"""Wave-972 free-probability-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.free_berg import bench_free_berg
from quant_fund.models.free_cumulant import bench_free_cumulant
from quant_fund.models.free_entropy import bench_free_entropy
from quant_fund.models.free_fisher_info import bench_free_fisher_info
from quant_fund.models.freeness_check import bench_freeness_check
from quant_fund.models.matrix_model_free import bench_matrix_model_free

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


def bench_free_entropy_family(seed: int = _SEED + 36000) -> dict[str, float]:
    return _finite_blob(bench_free_entropy(seed))


def bench_free_fisher_info_family(seed: int = _SEED + 36001) -> dict[str, float]:
    return _finite_blob(bench_free_fisher_info(seed))


def bench_free_cumulant_family(seed: int = _SEED + 36002) -> dict[str, float]:
    return _finite_blob(bench_free_cumulant(seed))


def bench_freeness_check_family(seed: int = _SEED + 36003) -> dict[str, float]:
    return _finite_blob(bench_freeness_check(seed))


def bench_matrix_model_free_family(seed: int = _SEED + 36004) -> dict[str, float]:
    return _finite_blob(bench_matrix_model_free(seed))


def bench_free_berg_family(seed: int = _SEED + 36005) -> dict[str, float]:
    return _finite_blob(bench_free_berg(seed))
