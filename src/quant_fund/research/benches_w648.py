"""Wave-648 algebraic-K-10 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dupont_k import bench_dupont_k
from quant_fund.models.guin_k import bench_guin_k
from quant_fund.models.kodaira_k import bench_kodaira_k
from quant_fund.models.lindenstrauss_k import (
    bench_lindenstrauss_k,
)
from quant_fund.models.suslin_k2 import bench_suslin_k2
from quant_fund.models.tsukada_k import bench_tsukada_k

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


def bench_kodaira_k_family(
    seed: int = _SEED + 3794,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kodaira_k",
            bench_kodaira_k(seed),
        )
    )


def bench_lindenstrauss_k_family(
    seed: int = _SEED + 3795,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lindenstrauss_k",
            bench_lindenstrauss_k(seed),
        )
    )


def bench_tsukada_k_family(
    seed: int = _SEED + 3796,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tsukada_k",
            bench_tsukada_k(seed),
        )
    )


def bench_guin_k_family(
    seed: int = _SEED + 3797,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "guin_k",
            bench_guin_k(seed),
        )
    )


def bench_dupont_k_family(
    seed: int = _SEED + 3798,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dupont_k",
            bench_dupont_k(seed),
        )
    )


def bench_suslin_k2_family(
    seed: int = _SEED + 3799,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "suslin_k2",
            bench_suslin_k2(seed),
        )
    )
