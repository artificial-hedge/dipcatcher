"""Wave-986 Calderon-Zygmund canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.ap_weight import bench_ap_weight
from quant_fund.models.calderon_zygmund import bench_calderon_zygmund
from quant_fund.models.cotlar_ineq import bench_cotlar_ineq
from quant_fund.models.cz_decomp import bench_cz_decomp
from quant_fund.models.good_lambda import bench_good_lambda
from quant_fund.models.reverse_holder import bench_reverse_holder

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


def bench_calderon_zygmund_family(seed: int = _SEED + 37400) -> dict[str, float]:
    return _finite_blob(bench_calderon_zygmund(seed))


def bench_cz_decomp_family(seed: int = _SEED + 37401) -> dict[str, float]:
    return _finite_blob(bench_cz_decomp(seed))


def bench_cotlar_ineq_family(seed: int = _SEED + 37402) -> dict[str, float]:
    return _finite_blob(bench_cotlar_ineq(seed))


def bench_good_lambda_family(seed: int = _SEED + 37403) -> dict[str, float]:
    return _finite_blob(bench_good_lambda(seed))


def bench_ap_weight_family(seed: int = _SEED + 37404) -> dict[str, float]:
    return _finite_blob(bench_ap_weight(seed))


def bench_reverse_holder_family(seed: int = _SEED + 37405) -> dict[str, float]:
    return _finite_blob(bench_reverse_holder(seed))
