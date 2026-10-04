"""Wave-569 geometric-PDE bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.aubin_thm import bench_aubin_thm
from quant_fund.models.kazdan_warner import bench_kazdan_warner
from quant_fund.models.nirenberg_problem import (
    bench_nirenberg_problem,
)
from quant_fund.models.prescribed_curvature import (
    bench_prescribed_curvature,
)
from quant_fund.models.trudinger_thm import bench_trudinger_thm
from quant_fund.models.yamabe_problem import bench_yamabe_problem

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


def bench_yamabe_problem_family(
    seed: int = _SEED + 3320,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "yamabe_problem",
            bench_yamabe_problem(seed),
        )
    )


def bench_prescribed_curvature_family(
    seed: int = _SEED + 3321,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "prescribed_curvature",
            bench_prescribed_curvature(seed),
        )
    )


def bench_nirenberg_problem_family(
    seed: int = _SEED + 3322,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nirenberg_problem",
            bench_nirenberg_problem(seed),
        )
    )


def bench_kazdan_warner_family(
    seed: int = _SEED + 3323,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kazdan_warner",
            bench_kazdan_warner(seed),
        )
    )


def bench_aubin_thm_family(seed: int = _SEED + 3324) -> dict[str, float]:
    return _floats(_finite_blob("aubin_thm", bench_aubin_thm(seed)))


def bench_trudinger_thm_family(
    seed: int = _SEED + 3325,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "trudinger_thm",
            bench_trudinger_thm(seed),
        )
    )
