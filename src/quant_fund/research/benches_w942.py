"""Wave-942 primal-dual canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.backward_forward import bench_backward_forward
from quant_fund.models.ishikawa_iter import bench_ishikawa_iter
from quant_fund.models.malitsky_golden import bench_malitsky_golden
from quant_fund.models.mann_iter import bench_mann_iter
from quant_fund.models.primal_dual_hybrid import bench_primal_dual_hybrid
from quant_fund.models.vu_condat import bench_vu_condat

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


def bench_primal_dual_hybrid_family(seed: int = _SEED + 33000) -> dict[str, float]:
    return _finite_blob(bench_primal_dual_hybrid(seed))


def bench_vu_condat_family(seed: int = _SEED + 33001) -> dict[str, float]:
    return _finite_blob(bench_vu_condat(seed))


def bench_backward_forward_family(seed: int = _SEED + 33002) -> dict[str, float]:
    return _finite_blob(bench_backward_forward(seed))


def bench_malitsky_golden_family(seed: int = _SEED + 33003) -> dict[str, float]:
    return _finite_blob(bench_malitsky_golden(seed))


def bench_mann_iter_family(seed: int = _SEED + 33004) -> dict[str, float]:
    return _finite_blob(bench_mann_iter(seed))


def bench_ishikawa_iter_family(seed: int = _SEED + 33005) -> dict[str, float]:
    return _finite_blob(bench_ishikawa_iter(seed))
