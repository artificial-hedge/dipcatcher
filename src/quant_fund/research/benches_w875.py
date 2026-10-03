"""Wave-875 adaptive-marking bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adaptive_finite import (
    bench_adaptive_finite,
)
from quant_fund.models.adaptive_marking import (
    bench_adaptive_marking,
)
from quant_fund.models.convergence_theory import (
    bench_convergence_theory,
)
from quant_fund.models.dorfler_marking import (
    bench_dorfler_marking,
)
from quant_fund.models.goal_adaptive import (
    bench_goal_adaptive,
)
from quant_fund.models.hierarchical_est import (
    bench_hierarchical_est,
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


def bench_adaptive_marking_family(
    seed: int = _SEED + 26300,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "adaptive_marking",
            bench_adaptive_marking(seed),
        )
    )


def bench_hierarchical_est_family(
    seed: int = _SEED + 26301,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hierarchical_est",
            bench_hierarchical_est(seed),
        )
    )


def bench_dorfler_marking_family(
    seed: int = _SEED + 26302,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dorfler_marking",
            bench_dorfler_marking(seed),
        )
    )


def bench_convergence_theory_family(
    seed: int = _SEED + 26303,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "convergence_theory",
            bench_convergence_theory(seed),
        )
    )


def bench_adaptive_finite_family(
    seed: int = _SEED + 26304,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "adaptive_finite",
            bench_adaptive_finite(seed),
        )
    )


def bench_goal_adaptive_family(
    seed: int = _SEED + 26305,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "goal_adaptive",
            bench_goal_adaptive(seed),
        )
    )
