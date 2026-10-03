"""Wave-878 low-rank/hierarchical-matrix bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.block_low_rank import (
    bench_block_low_rank,
)
from quant_fund.models.h_matrix import (
    bench_h_matrix,
)
from quant_fund.models.hss_matrix import (
    bench_hss_matrix,
)
from quant_fund.models.kronecker_approx import (
    bench_kronecker_approx,
)
from quant_fund.models.low_rank_svd import (
    bench_low_rank_svd,
)
from quant_fund.models.randomized_nystrom import (
    bench_randomized_nystrom,
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


def bench_low_rank_svd_family(
    seed: int = _SEED + 26600,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "low_rank_svd",
            bench_low_rank_svd(seed),
        )
    )


def bench_h_matrix_family(
    seed: int = _SEED + 26601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "h_matrix",
            bench_h_matrix(seed),
        )
    )


def bench_hss_matrix_family(
    seed: int = _SEED + 26602,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hss_matrix",
            bench_hss_matrix(seed),
        )
    )


def bench_randomized_nystrom_family(
    seed: int = _SEED + 26603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "randomized_nystrom",
            bench_randomized_nystrom(seed),
        )
    )


def bench_block_low_rank_family(
    seed: int = _SEED + 26604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "block_low_rank",
            bench_block_low_rank(seed),
        )
    )


def bench_kronecker_approx_family(
    seed: int = _SEED + 26605,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kronecker_approx",
            bench_kronecker_approx(seed),
        )
    )
