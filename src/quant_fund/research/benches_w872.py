"""Wave-872 preconditioner bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.amg_precond import (
    bench_amg_precond,
)
from quant_fund.models.ic_precond import (
    bench_ic_precond,
)
from quant_fund.models.ilut_precond import (
    bench_ilut_precond,
)
from quant_fund.models.jacobi_precond import (
    bench_jacobi_precond,
)
from quant_fund.models.polynomial_precond import (
    bench_polynomial_precond,
)
from quant_fund.models.ssor_precond import (
    bench_ssor_precond,
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


def bench_jacobi_precond_family(
    seed: int = _SEED + 26000,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "jacobi_precond",
            bench_jacobi_precond(seed),
        )
    )


def bench_ilut_precond_family(
    seed: int = _SEED + 26001,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ilut_precond",
            bench_ilut_precond(seed),
        )
    )


def bench_ssor_precond_family(
    seed: int = _SEED + 26002,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ssor_precond",
            bench_ssor_precond(seed),
        )
    )


def bench_amg_precond_family(
    seed: int = _SEED + 26003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "amg_precond",
            bench_amg_precond(seed),
        )
    )


def bench_ic_precond_family(
    seed: int = _SEED + 26004,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ic_precond",
            bench_ic_precond(seed),
        )
    )


def bench_polynomial_precond_family(
    seed: int = _SEED + 26005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "polynomial_precond",
            bench_polynomial_precond(seed),
        )
    )
