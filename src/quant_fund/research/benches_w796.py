"""Wave-796 FBSDE-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.coupled_fbsde import (
    bench_coupled_fbsde,
)
from quant_fund.models.decoupling_field2 import (
    bench_decoupling_field2,
)
from quant_fund.models.four_step_scheme import (
    bench_four_step_scheme,
)
from quant_fund.models.quasi_bsde import (
    bench_quasi_bsde,
)
from quant_fund.models.random_bsde import (
    bench_random_bsde,
)
from quant_fund.models.time_bsde import (
    bench_time_bsde,
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


def bench_four_step_scheme_family(
    seed: int = _SEED + 18500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "four_step_scheme",
            bench_four_step_scheme(seed),
        )
    )


def bench_decoupling_field2_family(
    seed: int = _SEED + 18501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "decoupling_field2",
            bench_decoupling_field2(seed),
        )
    )


def bench_quasi_bsde_family(
    seed: int = _SEED + 18502,
) -> dict[str, float]:
    return _floats(_finite_blob("quasi_bsde", bench_quasi_bsde(seed)))


def bench_coupled_fbsde_family(
    seed: int = _SEED + 18503,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "coupled_fbsde",
            bench_coupled_fbsde(seed),
        )
    )


def bench_random_bsde_family(
    seed: int = _SEED + 18504,
) -> dict[str, float]:
    return _floats(_finite_blob("random_bsde", bench_random_bsde(seed)))


def bench_time_bsde_family(
    seed: int = _SEED + 18505,
) -> dict[str, float]:
    return _floats(_finite_blob("time_bsde", bench_time_bsde(seed)))
