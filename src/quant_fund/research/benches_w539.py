"""Wave-539 diophantine-approximation bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.continued_frac2 import bench_continued_frac2
from quant_fund.models.dirichlet_approx import bench_dirichlet_approx
from quant_fund.models.kronecker_thm import bench_kronecker_thm
from quant_fund.models.liouville_number import bench_liouville_number
from quant_fund.models.roth_thm2 import bench_roth_thm2
from quant_fund.models.subspace_thm import bench_subspace_thm

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


def bench_dirichlet_approx_family(
    seed: int = _SEED + 3140,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dirichlet_approx",
            bench_dirichlet_approx(seed),
        )
    )


def bench_roth_thm2_family(seed: int = _SEED + 3141) -> dict[str, float]:
    return _floats(_finite_blob("roth_thm2", bench_roth_thm2(seed)))


def bench_continued_frac2_family(
    seed: int = _SEED + 3142,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "continued_frac2",
            bench_continued_frac2(seed),
        )
    )


def bench_kronecker_thm_family(seed: int = _SEED + 3143) -> dict[str, float]:
    return _floats(_finite_blob("kronecker_thm", bench_kronecker_thm(seed)))


def bench_liouville_number_family(
    seed: int = _SEED + 3144,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "liouville_number",
            bench_liouville_number(seed),
        )
    )


def bench_subspace_thm_family(seed: int = _SEED + 3145) -> dict[str, float]:
    return _floats(_finite_blob("subspace_thm", bench_subspace_thm(seed)))
