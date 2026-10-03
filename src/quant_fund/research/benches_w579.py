"""Wave-579 geometric-invariant-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.git_quotient import bench_git_quotient
from quant_fund.models.hilbert_mumford import (
    bench_hilbert_mumford,
)
from quant_fund.models.kirwan_strat import bench_kirwan_strat
from quant_fund.models.luna_slice import bench_luna_slice
from quant_fund.models.moment_polytope import (
    bench_moment_polytope,
)
from quant_fund.models.symplectic_quot import (
    bench_symplectic_quot,
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


def bench_git_quotient_family(seed: int = _SEED + 3380) -> dict[str, float]:
    return _floats(_finite_blob("git_quotient", bench_git_quotient(seed)))


def bench_hilbert_mumford_family(
    seed: int = _SEED + 3381,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hilbert_mumford",
            bench_hilbert_mumford(seed),
        )
    )


def bench_moment_polytope_family(
    seed: int = _SEED + 3382,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "moment_polytope",
            bench_moment_polytope(seed),
        )
    )


def bench_kirwan_strat_family(seed: int = _SEED + 3383) -> dict[str, float]:
    return _floats(_finite_blob("kirwan_strat", bench_kirwan_strat(seed)))


def bench_symplectic_quot_family(
    seed: int = _SEED + 3384,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "symplectic_quot",
            bench_symplectic_quot(seed),
        )
    )


def bench_luna_slice_family(seed: int = _SEED + 3385) -> dict[str, float]:
    return _floats(_finite_blob("luna_slice", bench_luna_slice(seed)))
