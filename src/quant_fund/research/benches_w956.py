"""Wave-956 tensor-theory canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.cp_rank import bench_cp_rank
from quant_fund.models.mode_n_product import bench_mode_n_product
from quant_fund.models.tensor_norm import bench_tensor_norm
from quant_fund.models.tensor_symmetry import bench_tensor_symmetry
from quant_fund.models.tensor_trace import bench_tensor_trace
from quant_fund.models.tucker_rank import bench_tucker_rank

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


def bench_tucker_rank_family(seed: int = _SEED + 34400) -> dict[str, float]:
    return _finite_blob(bench_tucker_rank(seed))


def bench_cp_rank_family(seed: int = _SEED + 34401) -> dict[str, float]:
    return _finite_blob(bench_cp_rank(seed))


def bench_tensor_norm_family(seed: int = _SEED + 34402) -> dict[str, float]:
    return _finite_blob(bench_tensor_norm(seed))


def bench_tensor_trace_family(seed: int = _SEED + 34403) -> dict[str, float]:
    return _finite_blob(bench_tensor_trace(seed))


def bench_mode_n_product_family(seed: int = _SEED + 34404) -> dict[str, float]:
    return _finite_blob(bench_mode_n_product(seed))


def bench_tensor_symmetry_family(seed: int = _SEED + 34405) -> dict[str, float]:
    return _finite_blob(bench_tensor_symmetry(seed))
