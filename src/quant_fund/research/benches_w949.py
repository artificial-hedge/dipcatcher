"""Wave-949 matrix-pencil canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.deflating_subspace import bench_deflating_subspace
from quant_fund.models.invariant_subspace import bench_invariant_subspace
from quant_fund.models.jordan_form import bench_jordan_form
from quant_fund.models.kronecker_canonical import bench_kronecker_canonical
from quant_fund.models.matrix_pencil import bench_matrix_pencil
from quant_fund.models.rational_canonical import bench_rational_canonical

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


def bench_matrix_pencil_family(seed: int = _SEED + 33700) -> dict[str, float]:
    return _finite_blob(bench_matrix_pencil(seed))


def bench_kronecker_canonical_family(seed: int = _SEED + 33701) -> dict[str, float]:
    return _finite_blob(bench_kronecker_canonical(seed))


def bench_invariant_subspace_family(seed: int = _SEED + 33702) -> dict[str, float]:
    return _finite_blob(bench_invariant_subspace(seed))


def bench_deflating_subspace_family(seed: int = _SEED + 33703) -> dict[str, float]:
    return _finite_blob(bench_deflating_subspace(seed))


def bench_jordan_form_family(seed: int = _SEED + 33704) -> dict[str, float]:
    return _finite_blob(bench_jordan_form(seed))


def bench_rational_canonical_family(seed: int = _SEED + 33705) -> dict[str, float]:
    return _finite_blob(bench_rational_canonical(seed))
