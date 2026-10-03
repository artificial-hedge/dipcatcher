"""Wave-876 nonlinear-solver bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.anderson_mixing import (
    bench_anderson_mixing,
)
from quant_fund.models.conjugate_grad_ls import (
    bench_conjugate_grad_ls,
)
from quant_fund.models.gauss_newton import (
    bench_gauss_newton,
)
from quant_fund.models.landweber_iter import (
    bench_landweber_iter,
)
from quant_fund.models.levenberg_marq import (
    bench_levenberg_marq,
)
from quant_fund.models.moore_penrose import (
    bench_moore_penrose,
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


def bench_moore_penrose_family(
    seed: int = _SEED + 26400,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "moore_penrose",
            bench_moore_penrose(seed),
        )
    )


def bench_landweber_iter_family(
    seed: int = _SEED + 26401,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "landweber_iter",
            bench_landweber_iter(seed),
        )
    )


def bench_conjugate_grad_ls_family(
    seed: int = _SEED + 26402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "conjugate_grad_ls",
            bench_conjugate_grad_ls(seed),
        )
    )


def bench_gauss_newton_family(
    seed: int = _SEED + 26403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gauss_newton",
            bench_gauss_newton(seed),
        )
    )


def bench_levenberg_marq_family(
    seed: int = _SEED + 26404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "levenberg_marq",
            bench_levenberg_marq(seed),
        )
    )


def bench_anderson_mixing_family(
    seed: int = _SEED + 26405,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "anderson_mixing",
            bench_anderson_mixing(seed),
        )
    )
