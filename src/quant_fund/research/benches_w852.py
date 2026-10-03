"""Wave-852 boundary-element bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bem_kernel import (
    bench_bem_kernel,
)
from quant_fund.models.fast_multipole import (
    bench_fast_multipole,
)
from quant_fund.models.fredholm_solve import (
    bench_fredholm_solve,
)
from quant_fund.models.galerkin_bem import (
    bench_galerkin_bem,
)
from quant_fund.models.nystrom_method import (
    bench_nystrom_method,
)
from quant_fund.models.singular_integrals import (
    bench_singular_integrals,
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


def bench_bem_kernel_family(
    seed: int = _SEED + 24000,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bem_kernel",
            bench_bem_kernel(seed),
        )
    )


def bench_fredholm_solve_family(
    seed: int = _SEED + 24001,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fredholm_solve",
            bench_fredholm_solve(seed),
        )
    )


def bench_nystrom_method_family(
    seed: int = _SEED + 24002,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nystrom_method",
            bench_nystrom_method(seed),
        )
    )


def bench_singular_integrals_family(
    seed: int = _SEED + 24003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "singular_integrals",
            bench_singular_integrals(seed),
        )
    )


def bench_fast_multipole_family(
    seed: int = _SEED + 24004,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fast_multipole",
            bench_fast_multipole(seed),
        )
    )


def bench_galerkin_bem_family(
    seed: int = _SEED + 24005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "galerkin_bem",
            bench_galerkin_bem(seed),
        )
    )
