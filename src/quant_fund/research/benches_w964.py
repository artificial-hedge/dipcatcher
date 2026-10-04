"""Wave-964 unbounded-operator canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.adjoint_unbounded import bench_adjoint_unbounded
from quant_fund.models.closed_operator import bench_closed_operator
from quant_fund.models.domain_dense import bench_domain_dense
from quant_fund.models.resolvent_op import bench_resolvent_op
from quant_fund.models.spectral_measure import bench_spectral_measure
from quant_fund.models.unbounded_operator import bench_unbounded_operator

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


def bench_unbounded_operator_family(seed: int = _SEED + 35200) -> dict[str, float]:
    return _finite_blob(bench_unbounded_operator(seed))


def bench_closed_operator_family(seed: int = _SEED + 35201) -> dict[str, float]:
    return _finite_blob(bench_closed_operator(seed))


def bench_domain_dense_family(seed: int = _SEED + 35202) -> dict[str, float]:
    return _finite_blob(bench_domain_dense(seed))


def bench_adjoint_unbounded_family(seed: int = _SEED + 35203) -> dict[str, float]:
    return _finite_blob(bench_adjoint_unbounded(seed))


def bench_resolvent_op_family(seed: int = _SEED + 35204) -> dict[str, float]:
    return _finite_blob(bench_resolvent_op(seed))


def bench_spectral_measure_family(seed: int = _SEED + 35205) -> dict[str, float]:
    return _finite_blob(bench_spectral_measure(seed))
