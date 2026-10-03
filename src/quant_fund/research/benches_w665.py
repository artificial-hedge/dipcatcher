"""Wave-665 homotopy-23 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cohen_moore2 import bench_cohen_moore2
from quant_fund.models.homotopy_decomp import bench_homotopy_decomp
from quant_fund.models.kervaire_inv2 import bench_kervaire_inv2
from quant_fund.models.moore_space2 import bench_moore_space2
from quant_fund.models.unstable_vn import bench_unstable_vn
from quant_fund.models.whitehead_product import (
    bench_whitehead_product,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_cohen_moore2_family(
    seed: int = _SEED + 5400,
) -> dict[str, float]:
    return _floats(_finite_blob("cohen_moore2", bench_cohen_moore2(seed)))


def bench_whitehead_product_family(
    seed: int = _SEED + 5401,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "whitehead_product",
            bench_whitehead_product(seed),
        )
    )


def bench_homotopy_decomp_family(
    seed: int = _SEED + 5402,
) -> dict[str, float]:
    return _floats(_finite_blob("homotopy_decomp", bench_homotopy_decomp(seed)))


def bench_kervaire_inv2_family(
    seed: int = _SEED + 5403,
) -> dict[str, float]:
    return _floats(_finite_blob("kervaire_inv2", bench_kervaire_inv2(seed)))


def bench_unstable_vn_family(
    seed: int = _SEED + 5404,
) -> dict[str, float]:
    return _floats(_finite_blob("unstable_vn", bench_unstable_vn(seed)))


def bench_moore_space2_family(
    seed: int = _SEED + 5405,
) -> dict[str, float]:
    return _floats(_finite_blob("moore_space2", bench_moore_space2(seed)))
