"""Wave-126 adapters: exec-summary differentiable-algorithm canon — ode_adjoint,
st_estimator, implicit_diff, gumbel_relax, perturb_map, smooth_argmax —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.gumbel_relax import bench_gumbel_relax
from quant_fund.models.implicit_diff import bench_implicit_diff
from quant_fund.models.ode_adjoint import bench_ode_adjoint
from quant_fund.models.perturb_map import bench_perturb_map
from quant_fund.models.smooth_argmax import bench_smooth_argmax
from quant_fund.models.st_estimator import bench_st_estimator

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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


def bench_ode_adjoint_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ode_adjoint", bench_ode_adjoint(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ode_adjoint bench failed: {exc}") from exc


def bench_st_estimator_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("st_estimator", bench_st_estimator(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"st_estimator bench failed: {exc}") from exc


def bench_implicit_diff_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("implicit_diff", bench_implicit_diff(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"implicit_diff bench failed: {exc}") from exc


def bench_gumbel_relax_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gumbel_relax", bench_gumbel_relax(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gumbel_relax bench failed: {exc}") from exc


def bench_perturb_map_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("perturb_map", bench_perturb_map(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"perturb_map bench failed: {exc}") from exc


def bench_smooth_argmax_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("smooth_argmax", bench_smooth_argmax(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"smooth_argmax bench failed: {exc}") from exc
