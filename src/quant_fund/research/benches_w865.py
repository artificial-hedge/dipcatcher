"""Wave-865 a-posteriori error-estimation bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dual_weighted_res import (
    bench_dual_weighted_res,
)
from quant_fund.models.equilibrated_flux import (
    bench_equilibrated_flux,
)
from quant_fund.models.goal_oriented import (
    bench_goal_oriented,
)
from quant_fund.models.recovery_error import (
    bench_recovery_error,
)
from quant_fund.models.residual_estimator import (
    bench_residual_estimator,
)
from quant_fund.models.zienkiewicz_zhu import (
    bench_zienkiewicz_zhu,
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


def bench_residual_estimator_family(
    seed: int = _SEED + 25300,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "residual_estimator",
            bench_residual_estimator(seed),
        )
    )


def bench_zienkiewicz_zhu_family(
    seed: int = _SEED + 25301,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "zienkiewicz_zhu",
            bench_zienkiewicz_zhu(seed),
        )
    )


def bench_recovery_error_family(
    seed: int = _SEED + 25302,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "recovery_error",
            bench_recovery_error(seed),
        )
    )


def bench_dual_weighted_res_family(
    seed: int = _SEED + 25303,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dual_weighted_res",
            bench_dual_weighted_res(seed),
        )
    )


def bench_goal_oriented_family(
    seed: int = _SEED + 25304,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "goal_oriented",
            bench_goal_oriented(seed),
        )
    )


def bench_equilibrated_flux_family(
    seed: int = _SEED + 25305,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "equilibrated_flux",
            bench_equilibrated_flux(seed),
        )
    )
