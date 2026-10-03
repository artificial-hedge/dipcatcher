"""Wave-645 algebraic-K-9 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.balmer_k import bench_balmer_k
from quant_fund.models.hermitian_k3 import (
    bench_hermitian_k3,
)
from quant_fund.models.schlichting_k import (
    bench_schlichting_k,
)
from quant_fund.models.thomason_les import (
    bench_thomason_les,
)
from quant_fund.models.vishik_k import bench_vishik_k
from quant_fund.models.witt_k import bench_witt_k

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


def bench_witt_k_family(
    seed: int = _SEED + 3776,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "witt_k",
            bench_witt_k(seed),
        )
    )


def bench_schlichting_k_family(
    seed: int = _SEED + 3777,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "schlichting_k",
            bench_schlichting_k(seed),
        )
    )


def bench_balmer_k_family(
    seed: int = _SEED + 3778,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "balmer_k",
            bench_balmer_k(seed),
        )
    )


def bench_hermitian_k3_family(
    seed: int = _SEED + 3779,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hermitian_k3",
            bench_hermitian_k3(seed),
        )
    )


def bench_thomason_les_family(
    seed: int = _SEED + 3780,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "thomason_les",
            bench_thomason_les(seed),
        )
    )


def bench_vishik_k_family(
    seed: int = _SEED + 3781,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "vishik_k",
            bench_vishik_k(seed),
        )
    )
