"""Wave-950 tensor-algebra canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.hadamard_product import bench_hadamard_product
from quant_fund.models.khatri_rao import bench_khatri_rao
from quant_fund.models.kron_product import bench_kron_product
from quant_fund.models.outer_product import bench_outer_product
from quant_fund.models.tensor_contraction import bench_tensor_contraction
from quant_fund.models.tensor_unfold import bench_tensor_unfold

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


def bench_tensor_contraction_family(seed: int = _SEED + 33800) -> dict[str, float]:
    return _finite_blob(bench_tensor_contraction(seed))


def bench_khatri_rao_family(seed: int = _SEED + 33801) -> dict[str, float]:
    return _finite_blob(bench_khatri_rao(seed))


def bench_kron_product_family(seed: int = _SEED + 33802) -> dict[str, float]:
    return _finite_blob(bench_kron_product(seed))


def bench_hadamard_product_family(seed: int = _SEED + 33803) -> dict[str, float]:
    return _finite_blob(bench_hadamard_product(seed))


def bench_tensor_unfold_family(seed: int = _SEED + 33804) -> dict[str, float]:
    return _finite_blob(bench_tensor_unfold(seed))


def bench_outer_product_family(seed: int = _SEED + 33805) -> dict[str, float]:
    return _finite_blob(bench_outer_product(seed))
