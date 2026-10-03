"""Wave-893 collocation/least-squares bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.coarsening_mark import (
    bench_coarsening_mark,
)
from quant_fund.models.covello_est import (
    bench_covello_est,
)
from quant_fund.models.dual_goal_est import (
    bench_dual_goal_est,
)
from quant_fund.models.galerkin_least_sq import (
    bench_galerkin_least_sq,
)
from quant_fund.models.pseudospectral_coll import (
    bench_pseudospectral_coll,
)
from quant_fund.models.tau_method import (
    bench_tau_method,
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


def bench_covello_est_family(
    seed: int = _SEED + 28100,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "covello_est",
            bench_covello_est(seed),
        )
    )


def bench_dual_goal_est_family(
    seed: int = _SEED + 28101,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dual_goal_est",
            bench_dual_goal_est(seed),
        )
    )


def bench_pseudospectral_coll_family(
    seed: int = _SEED + 28102,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "pseudospectral_coll",
            bench_pseudospectral_coll(seed),
        )
    )


def bench_tau_method_family(
    seed: int = _SEED + 28103,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tau_method",
            bench_tau_method(seed),
        )
    )


def bench_galerkin_least_sq_family(
    seed: int = _SEED + 28104,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "galerkin_least_sq",
            bench_galerkin_least_sq(seed),
        )
    )


def bench_coarsening_mark_family(
    seed: int = _SEED + 28105,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "coarsening_mark",
            bench_coarsening_mark(seed),
        )
    )
