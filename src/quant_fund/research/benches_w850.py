"""Wave-850 Riemann-solver bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ausm_flux import (
    bench_ausm_flux,
)
from quant_fund.models.godunov_exact import (
    bench_godunov_exact,
)
from quant_fund.models.hllc_solver import (
    bench_hllc_solver,
)
from quant_fund.models.lax_friedrichs import (
    bench_lax_friedrichs,
)
from quant_fund.models.osher_solver import (
    bench_osher_solver,
)
from quant_fund.models.roe_solver import (
    bench_roe_solver,
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


def bench_roe_solver_family(
    seed: int = _SEED + 23800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "roe_solver",
            bench_roe_solver(seed),
        )
    )


def bench_hllc_solver_family(
    seed: int = _SEED + 23801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hllc_solver",
            bench_hllc_solver(seed),
        )
    )


def bench_ausm_flux_family(
    seed: int = _SEED + 23802,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ausm_flux",
            bench_ausm_flux(seed),
        )
    )


def bench_lax_friedrichs_family(
    seed: int = _SEED + 23803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lax_friedrichs",
            bench_lax_friedrichs(seed),
        )
    )


def bench_godunov_exact_family(
    seed: int = _SEED + 23804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "godunov_exact",
            bench_godunov_exact(seed),
        )
    )


def bench_osher_solver_family(
    seed: int = _SEED + 23805,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "osher_solver",
            bench_osher_solver(seed),
        )
    )
