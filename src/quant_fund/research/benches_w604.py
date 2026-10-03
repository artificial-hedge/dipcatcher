"""Wave-604 homotopy-12 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bokstedt_periodicity import (
    bench_bokstedt_periodicity,
)
from quant_fund.models.elliptic_k import bench_elliptic_k
from quant_fund.models.equivariant_cohomology2 import (
    bench_equivariant_cohomology2,
)
from quant_fund.models.may_ss import bench_may_ss
from quant_fund.models.topo_k_theory import (
    bench_topo_k_theory,
)
from quant_fund.models.unstable_cohomology import (
    bench_unstable_cohomology,
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


def bench_unstable_cohomology_family(
    seed: int = _SEED + 3530,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "unstable_cohomology",
            bench_unstable_cohomology(seed),
        )
    )


def bench_may_ss_family(seed: int = _SEED + 3531) -> dict[str, float]:
    return _floats(_finite_blob("may_ss", bench_may_ss(seed)))


def bench_bokstedt_periodicity_family(
    seed: int = _SEED + 3532,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bokstedt_periodicity",
            bench_bokstedt_periodicity(seed),
        )
    )


def bench_topo_k_theory_family(
    seed: int = _SEED + 3533,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "topo_k_theory",
            bench_topo_k_theory(seed),
        )
    )


def bench_elliptic_k_family(
    seed: int = _SEED + 3534,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "elliptic_k",
            bench_elliptic_k(seed),
        )
    )


def bench_equivariant_cohomology2_family(
    seed: int = _SEED + 3535,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "equivariant_cohomology2",
            bench_equivariant_cohomology2(seed),
        )
    )
