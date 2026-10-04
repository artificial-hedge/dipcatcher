"""Wave-943 matrix-analysis canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.gershgorin_disc import bench_gershgorin_disc
from quant_fund.models.kadison_ineq import bench_kadison_ineq
from quant_fund.models.loewner_matrix import bench_loewner_matrix
from quant_fund.models.operator_convex import bench_operator_convex
from quant_fund.models.ostrowski_bound import bench_ostrowski_bound
from quant_fund.models.wielandt_ineq import bench_wielandt_ineq

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


def bench_loewner_matrix_family(seed: int = _SEED + 33100) -> dict[str, float]:
    return _finite_blob(bench_loewner_matrix(seed))


def bench_operator_convex_family(seed: int = _SEED + 33101) -> dict[str, float]:
    return _finite_blob(bench_operator_convex(seed))


def bench_kadison_ineq_family(seed: int = _SEED + 33102) -> dict[str, float]:
    return _finite_blob(bench_kadison_ineq(seed))


def bench_wielandt_ineq_family(seed: int = _SEED + 33103) -> dict[str, float]:
    return _finite_blob(bench_wielandt_ineq(seed))


def bench_ostrowski_bound_family(seed: int = _SEED + 33104) -> dict[str, float]:
    return _finite_blob(bench_ostrowski_bound(seed))


def bench_gershgorin_disc_family(seed: int = _SEED + 33105) -> dict[str, float]:
    return _finite_blob(bench_gershgorin_disc(seed))
