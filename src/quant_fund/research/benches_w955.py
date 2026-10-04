"""Wave-955 linear-systems canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.back_substitution import bench_back_substitution
from quant_fund.models.forward_substitution import bench_forward_substitution
from quant_fund.models.givens_rotation import bench_givens_rotation
from quant_fund.models.gram_determinant import bench_gram_determinant
from quant_fund.models.gram_matrix import bench_gram_matrix
from quant_fund.models.householder_reflect import bench_householder_reflect

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


def bench_gram_matrix_family(seed: int = _SEED + 34300) -> dict[str, float]:
    return _finite_blob(bench_gram_matrix(seed))


def bench_gram_determinant_family(seed: int = _SEED + 34301) -> dict[str, float]:
    return _finite_blob(bench_gram_determinant(seed))


def bench_householder_reflect_family(seed: int = _SEED + 34302) -> dict[str, float]:
    return _finite_blob(bench_householder_reflect(seed))


def bench_givens_rotation_family(seed: int = _SEED + 34303) -> dict[str, float]:
    return _finite_blob(bench_givens_rotation(seed))


def bench_back_substitution_family(seed: int = _SEED + 34304) -> dict[str, float]:
    return _finite_blob(bench_back_substitution(seed))


def bench_forward_substitution_family(seed: int = _SEED + 34305) -> dict[str, float]:
    return _finite_blob(bench_forward_substitution(seed))
