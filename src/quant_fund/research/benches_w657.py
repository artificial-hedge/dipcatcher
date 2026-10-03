"""Wave-657 homotopy-22 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bousfield_period import bench_bousfield_period
from quant_fund.models.completion_htpy import bench_completion_htpy
from quant_fund.models.homotopy_cartesian import bench_homotopy_cartesian
from quant_fund.models.p_local_htpy import bench_p_local_htpy
from quant_fund.models.ravenel_htpy import bench_ravenel_htpy
from quant_fund.models.snake_constr import bench_snake_constr

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


def bench_ravenel_htpy_family(
    seed: int = _SEED + 4600,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ravenel_htpy",
            bench_ravenel_htpy(seed),
        )
    )


def bench_bousfield_period_family(
    seed: int = _SEED + 4601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bousfield_period",
            bench_bousfield_period(seed),
        )
    )


def bench_snake_constr_family(
    seed: int = _SEED + 4602,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "snake_constr",
            bench_snake_constr(seed),
        )
    )


def bench_homotopy_cartesian_family(
    seed: int = _SEED + 4603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "homotopy_cartesian",
            bench_homotopy_cartesian(seed),
        )
    )


def bench_p_local_htpy_family(
    seed: int = _SEED + 4604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "p_local_htpy",
            bench_p_local_htpy(seed),
        )
    )


def bench_completion_htpy_family(
    seed: int = _SEED + 4605,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "completion_htpy",
            bench_completion_htpy(seed),
        )
    )
