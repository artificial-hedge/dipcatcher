"""Wave-792 BSDE bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.backward_sde import (
    bench_backward_sde,
)
from quant_fund.models.bsde_solver import (
    bench_bsde_solver,
)
from quant_fund.models.fbsde_markov import (
    bench_fbsde_markov,
)
from quant_fund.models.pardoux_peng import (
    bench_pardoux_peng,
)
from quant_fund.models.reflected_bsde import (
    bench_reflected_bsde,
)
from quant_fund.models.second_order_bsde import (
    bench_second_order_bsde,
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


def bench_bsde_solver_family(
    seed: int = _SEED + 18100,
) -> dict[str, float]:
    return _floats(_finite_blob("bsde_solver", bench_bsde_solver(seed)))


def bench_fbsde_markov_family(
    seed: int = _SEED + 18101,
) -> dict[str, float]:
    return _floats(_finite_blob("fbsde_markov", bench_fbsde_markov(seed)))


def bench_backward_sde_family(
    seed: int = _SEED + 18102,
) -> dict[str, float]:
    return _floats(_finite_blob("backward_sde", bench_backward_sde(seed)))


def bench_pardoux_peng_family(
    seed: int = _SEED + 18103,
) -> dict[str, float]:
    return _floats(_finite_blob("pardoux_peng", bench_pardoux_peng(seed)))


def bench_reflected_bsde_family(
    seed: int = _SEED + 18104,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "reflected_bsde",
            bench_reflected_bsde(seed),
        )
    )


def bench_second_order_bsde_family(
    seed: int = _SEED + 18105,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "second_order_bsde",
            bench_second_order_bsde(seed),
        )
    )
