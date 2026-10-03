"""Wave-851 discontinuous-Galerkin bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dg_discretization import (
    bench_dg_discretization,
)
from quant_fund.models.limiter_tvb import (
    bench_limiter_tvb,
)
from quant_fund.models.modal_basis import (
    bench_modal_basis,
)
from quant_fund.models.numerical_flux_dg import (
    bench_numerical_flux_dg,
)
from quant_fund.models.penalty_dg import (
    bench_penalty_dg,
)
from quant_fund.models.rkdg_step import (
    bench_rkdg_step,
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


def bench_dg_discretization_family(
    seed: int = _SEED + 23900,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dg_discretization",
            bench_dg_discretization(seed),
        )
    )


def bench_numerical_flux_dg_family(
    seed: int = _SEED + 23901,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "numerical_flux_dg",
            bench_numerical_flux_dg(seed),
        )
    )


def bench_penalty_dg_family(
    seed: int = _SEED + 23902,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "penalty_dg",
            bench_penalty_dg(seed),
        )
    )


def bench_modal_basis_family(
    seed: int = _SEED + 23903,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "modal_basis",
            bench_modal_basis(seed),
        )
    )


def bench_limiter_tvb_family(
    seed: int = _SEED + 23904,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "limiter_tvb",
            bench_limiter_tvb(seed),
        )
    )


def bench_rkdg_step_family(
    seed: int = _SEED + 23905,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "rkdg_step",
            bench_rkdg_step(seed),
        )
    )
