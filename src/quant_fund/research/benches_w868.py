"""Wave-868 continuation/homotopy bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.arc_continuation import (
    bench_arc_continuation,
)
from quant_fund.models.bifurcation_track import (
    bench_bifurcation_track,
)
from quant_fund.models.davidenko_ode import (
    bench_davidenko_ode,
)
from quant_fund.models.deflation_method import (
    bench_deflation_method,
)
from quant_fund.models.homotopy_solver import (
    bench_homotopy_solver,
)
from quant_fund.models.pseudo_arclength import (
    bench_pseudo_arclength,
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


def bench_arc_continuation_family(
    seed: int = _SEED + 25600,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "arc_continuation",
            bench_arc_continuation(seed),
        )
    )


def bench_pseudo_arclength_family(
    seed: int = _SEED + 25601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "pseudo_arclength",
            bench_pseudo_arclength(seed),
        )
    )


def bench_deflation_method_family(
    seed: int = _SEED + 25602,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "deflation_method",
            bench_deflation_method(seed),
        )
    )


def bench_bifurcation_track_family(
    seed: int = _SEED + 25603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bifurcation_track",
            bench_bifurcation_track(seed),
        )
    )


def bench_homotopy_solver_family(
    seed: int = _SEED + 25604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "homotopy_solver",
            bench_homotopy_solver(seed),
        )
    )


def bench_davidenko_ode_family(
    seed: int = _SEED + 25605,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "davidenko_ode",
            bench_davidenko_ode(seed),
        )
    )
