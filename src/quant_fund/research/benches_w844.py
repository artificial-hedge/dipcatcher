"""Wave-844 asymptotic-analysis bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.asymptotic_series import (
    bench_asymptotic_series,
)
from quant_fund.models.borel_resum import (
    bench_borel_resum,
)
from quant_fund.models.poincare_expansion import (
    bench_poincare_expansion,
)
from quant_fund.models.stationary_phase import (
    bench_stationary_phase,
)
from quant_fund.models.steepest_descent import (
    bench_steepest_descent,
)
from quant_fund.models.wkb_approx import (
    bench_wkb_approx,
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


def bench_asymptotic_series_family(
    seed: int = _SEED + 23200,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "asymptotic_series",
            bench_asymptotic_series(seed),
        )
    )


def bench_poincare_expansion_family(
    seed: int = _SEED + 23201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "poincare_expansion",
            bench_poincare_expansion(seed),
        )
    )


def bench_steepest_descent_family(
    seed: int = _SEED + 23202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "steepest_descent",
            bench_steepest_descent(seed),
        )
    )


def bench_stationary_phase_family(
    seed: int = _SEED + 23203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stationary_phase",
            bench_stationary_phase(seed),
        )
    )


def bench_borel_resum_family(
    seed: int = _SEED + 23204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "borel_resum",
            bench_borel_resum(seed),
        )
    )


def bench_wkb_approx_family(
    seed: int = _SEED + 23205,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "wkb_approx",
            bench_wkb_approx(seed),
        )
    )
