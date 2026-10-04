"""Wave-864 stochastic-Galerkin/UQ bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.intrusive_pce import (
    bench_intrusive_pce,
)
from quant_fund.models.nonintrusive_pce import (
    bench_nonintrusive_pce,
)
from quant_fund.models.poly_chaos_uq import (
    bench_poly_chaos_uq,
)
from quant_fund.models.stochastic_colloc import (
    bench_stochastic_colloc,
)
from quant_fund.models.stochastic_fem import (
    bench_stochastic_fem,
)
from quant_fund.models.stochastic_galerkin import (
    bench_stochastic_galerkin,
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


def bench_stochastic_galerkin_family(
    seed: int = _SEED + 25200,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stochastic_galerkin",
            bench_stochastic_galerkin(seed),
        )
    )


def bench_poly_chaos_uq_family(
    seed: int = _SEED + 25201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "poly_chaos_uq",
            bench_poly_chaos_uq(seed),
        )
    )


def bench_intrusive_pce_family(
    seed: int = _SEED + 25202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "intrusive_pce",
            bench_intrusive_pce(seed),
        )
    )


def bench_nonintrusive_pce_family(
    seed: int = _SEED + 25203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nonintrusive_pce",
            bench_nonintrusive_pce(seed),
        )
    )


def bench_stochastic_colloc_family(
    seed: int = _SEED + 25204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stochastic_colloc",
            bench_stochastic_colloc(seed),
        )
    )


def bench_stochastic_fem_family(
    seed: int = _SEED + 25205,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stochastic_fem",
            bench_stochastic_fem(seed),
        )
    )
